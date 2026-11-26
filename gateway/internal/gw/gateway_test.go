package gw

import (
	"context"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/alireza-taheriF/pvap/gateway/internal/bridge"
	"github.com/alireza-taheriF/pvap/gateway/internal/g711"
	"github.com/alireza-taheriF/pvap/gateway/internal/rtp"
	"github.com/alireza-taheriF/pvap/gateway/internal/sip"
)

// fakeEngine accepts one call, records inbound audio, and runs a script.
type fakeEngine struct {
	ln      net.Listener
	start   chan string
	inbound chan []int16
	hangup  chan struct{}
	conn    chan net.Conn
}

func newFakeEngine(t *testing.T) *fakeEngine {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	e := &fakeEngine{ln: ln, start: make(chan string, 1), inbound: make(chan []int16, 1000),
		hangup: make(chan struct{}, 1), conn: make(chan net.Conn, 1)}
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		e.conn <- c
		for {
			kind, p, err := bridge.Read(c)
			if err != nil {
				return
			}
			switch kind {
			case bridge.Start:
				e.start <- string(p)
			case bridge.Audio:
				e.inbound <- bridge.BytesToPCM(p)
			case bridge.Hangup:
				e.hangup <- struct{}{}
				return
			}
		}
	}()
	return e
}

// ua is a bare-bones SIP user agent for tests.
type ua struct {
	t   *testing.T
	sip *net.UDPConn
	rtp *net.UDPConn
	gw  *net.UDPAddr
}

func newUA(t *testing.T, gw *net.UDPAddr) *ua {
	s, _ := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	r, _ := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	return &ua{t: t, sip: s, rtp: r, gw: gw}
}

func (u *ua) sendSIP(raw string) { u.sip.WriteToUDP([]byte(raw), u.gw) }

func (u *ua) recvSIP(timeout time.Duration) *sip.Message {
	u.sip.SetReadDeadline(time.Now().Add(timeout))
	buf := make([]byte, 65535)
	n, _, err := u.sip.ReadFromUDP(buf)
	if err != nil {
		u.t.Fatalf("no SIP message: %v", err)
	}
	m, err := sip.Parse(buf[:n])
	if err != nil {
		u.t.Fatal(err)
	}
	return m
}

func (u *ua) invite(callID string, pts string) string {
	sp := u.sip.LocalAddr().(*net.UDPAddr).Port
	rp := u.rtp.LocalAddr().(*net.UDPAddr).Port
	sdp := fmt.Sprintf("v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=audio %d RTP/AVP %s\r\n", rp, pts)
	return fmt.Sprintf("INVITE sip:agent@127.0.0.1 SIP/2.0\r\n"+
		"Via: SIP/2.0/UDP 127.0.0.1:%d;branch=z9hG4bK-inv\r\nMax-Forwards: 70\r\n"+
		"From: <sip:caller@127.0.0.1>;tag=ua1\r\nTo: <sip:agent@127.0.0.1>\r\n"+
		"Call-ID: %s\r\nCSeq: 1 INVITE\r\nContact: <sip:caller@127.0.0.1:%d>\r\n"+
		"Content-Type: application/sdp\r\nContent-Length: %d\r\n\r\n%s", sp, callID, sp, len(sdp), sdp)
}

func (u *ua) inDialog(method, callID, toTag string, cseq int) string {
	sp := u.sip.LocalAddr().(*net.UDPAddr).Port
	return fmt.Sprintf("%s sip:pvap@127.0.0.1 SIP/2.0\r\n"+
		"Via: SIP/2.0/UDP 127.0.0.1:%d;branch=z9hG4bK-%s%d\r\nMax-Forwards: 70\r\n"+
		"From: <sip:caller@127.0.0.1>;tag=ua1\r\nTo: <sip:agent@127.0.0.1>;tag=%s\r\n"+
		"Call-ID: %s\r\nCSeq: %d %s\r\nContent-Length: 0\r\n\r\n", method, sp, method, cseq, toTag, callID, cseq, method)
}

func startGateway(t *testing.T, engine string) (*Server, context.CancelFunc) {
	s, err := New(Config{SIPAddr: "127.0.0.1:0", PublicIP: "127.0.0.1", RTPPortMin: 31000, RTPPortMax: 31999,
		EngineAddr: engine, Logger: log.New(testLogOut, "", 0)})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go s.Serve(ctx)
	return s, cancel
}

func tone(n int, hz float64, start int) []int16 {
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(8000 * math.Sin(2*math.Pi*hz*float64(start+i)/8000))
	}
	return out
}

func answer(t *testing.T, u *ua, callID string) (*sip.Message, *net.UDPAddr) {
	u.sendSIP(u.invite(callID, "0 101"))
	if r := u.recvSIP(time.Second); r.StatusCode != 100 {
		t.Fatalf("want 100, got %d", r.StatusCode)
	}
	ok := u.recvSIP(2 * time.Second)
	if ok.StatusCode != 200 || ok.Get("Content-Type") != "application/sdp" {
		t.Fatalf("want 200 with SDP, got %d", ok.StatusCode)
	}
	a, err := sip.ParseSDP(ok.Body)
	if err != nil || a.PayloadTypes[0] != 0 {
		t.Fatalf("answer sdp %v %+v", err, a)
	}
	return ok, &net.UDPAddr{IP: net.ParseIP(a.ConnIP), Port: a.Port}
}

func TestCallMediaBothWaysAndCallerHangup(t *testing.T) {
	eng := newFakeEngine(t)
	s, cancel := startGateway(t, eng.ln.Addr().String())
	defer cancel()
	u := newUA(t, s.Addr())

	ok, gwRTP := answer(t, u, "call-1")
	tag := sip.Param(ok.Get("To"), "tag")
	u.sendSIP(u.inDialog("ACK", "call-1", tag, 1))

	select {
	case meta := <-eng.start:
		if !strings.Contains(meta, `"codec":"PCMU"`) || !strings.Contains(meta, `"call_id":"call-1"`) {
			t.Fatalf("start meta %s", meta)
		}
	case <-time.After(time.Second):
		t.Fatal("engine got no START")
	}

	// caller → engine: 10 frames of a 400 Hz tone, μ-law encoded
	for i := 0; i < 10; i++ {
		pcm := tone(160, 400, i*160)
		p := rtp.Packet{PayloadType: 0, Seq: uint16(i), Timestamp: uint32(i * 160), SSRC: 7, Payload: g711.PCMU.EncodeFrame(pcm, nil)}
		u.rtp.WriteToUDP(p.Marshal(), gwRTP)
	}
	for i := 0; i < 10; i++ {
		select {
		case got := <-eng.inbound:
			want := tone(160, 400, i*160)
			for j := range got {
				if d := int(got[j]) - int(want[j]); d > 300 || d < -300 {
					t.Fatalf("frame %d sample %d: got %d want %d", i, j, got[j], want[j])
				}
			}
		case <-time.After(time.Second):
			t.Fatalf("engine missing frame %d", i)
		}
	}

	// engine → caller: 15 frames; expect paced RTP with consecutive seq/ts
	ec := <-eng.conn
	for i := 0; i < 15; i++ {
		bridge.Write(ec, bridge.Audio, bridge.PCMToBytes(tone(160, 1000, i*160)))
	}
	var pkts []*rtp.Packet
	var arrivals []time.Time
	buf := make([]byte, 2048)
	deadline := time.Now().Add(2 * time.Second)
	for len(pkts) < 15 && time.Now().Before(deadline) {
		u.rtp.SetReadDeadline(deadline)
		n, _, err := u.rtp.ReadFromUDP(buf)
		if err != nil {
			break
		}
		p, _ := rtp.Unmarshal(buf[:n])
		if p.Marker || len(pkts) > 0 { // skip silence before the talkspurt
			pkts = append(pkts, p)
			arrivals = append(arrivals, time.Now())
		}
	}
	if len(pkts) != 15 {
		t.Fatalf("got %d agent RTP packets", len(pkts))
	}
	for i := 1; i < len(pkts); i++ {
		if pkts[i].Seq != pkts[i-1].Seq+1 || pkts[i].Timestamp != pkts[i-1].Timestamp+160 {
			t.Fatalf("seq/ts not consecutive at %d", i)
		}
	}
	if span := arrivals[14].Sub(arrivals[0]); span < 240*time.Millisecond || span > 400*time.Millisecond {
		t.Fatalf("not paced at 20 ms: 15 packets over %v", span)
	}
	if d := int(g711.UlawDecode(pkts[0].Payload[40])) - int(tone(160, 1000, 0)[40]); d > 300 || d < -300 {
		t.Fatal("agent audio content mismatch")
	}

	// caller hangs up
	u.sendSIP(u.inDialog("BYE", "call-1", tag, 2))
	if r := u.recvSIP(time.Second); r.StatusCode != 200 {
		t.Fatalf("BYE answered %d", r.StatusCode)
	}
	select {
	case <-eng.hangup:
	case <-time.After(time.Second):
		t.Fatal("engine not told about hangup")
	}
	time.Sleep(50 * time.Millisecond)
	if s.ActiveCalls() != 0 {
		t.Fatal("call not cleaned up")
	}
}

func TestClearFlushesPlayout(t *testing.T) {
	eng := newFakeEngine(t)
	s, cancel := startGateway(t, eng.ln.Addr().String())
	defer cancel()
	u := newUA(t, s.Addr())
	ok, _ := answer(t, u, "call-2")
	u.sendSIP(u.inDialog("ACK", "call-2", sip.Param(ok.Get("To"), "tag"), 1))
	<-eng.start
	ec := <-eng.conn
	for i := 0; i < 100; i++ { // 2 s of agent audio
		bridge.Write(ec, bridge.Audio, bridge.PCMToBytes(tone(160, 1000, i*160)))
	}
	time.Sleep(200 * time.Millisecond)
	bridge.Write(ec, bridge.Clear, nil)
	time.Sleep(100 * time.Millisecond)
	// after CLEAR, only silence should arrive
	buf := make([]byte, 2048)
	loud := 0
	end := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(end) {
		u.rtp.SetReadDeadline(end)
		n, _, err := u.rtp.ReadFromUDP(buf)
		if err != nil {
			break
		}
		p, _ := rtp.Unmarshal(buf[:n])
		if g711.UlawDecode(p.Payload[40]) != 0 || g711.UlawDecode(p.Payload[80]) != 0 {
			loud++
		}
	}
	if loud > 1 {
		t.Fatalf("%d non-silent packets after CLEAR", loud)
	}
}

func TestEngineHangupSendsBye(t *testing.T) {
	eng := newFakeEngine(t)
	s, cancel := startGateway(t, eng.ln.Addr().String())
	defer cancel()
	u := newUA(t, s.Addr())
	ok, _ := answer(t, u, "call-3")
	u.sendSIP(u.inDialog("ACK", "call-3", sip.Param(ok.Get("To"), "tag"), 1))
	<-eng.start
	ec := <-eng.conn
	bridge.Write(ec, bridge.Hangup, nil)
	bye := u.recvSIP(time.Second)
	if !bye.IsRequest || bye.Method != "BYE" || bye.Get("Call-ID") != "call-3" ||
		sip.Param(bye.Get("To"), "tag") != "ua1" || sip.Param(bye.Get("From"), "tag") == "" {
		t.Fatalf("bad BYE: %s", bye.Bytes())
	}
}

func TestRejectsUnsupportedCodecAndUnknownDialog(t *testing.T) {
	eng := newFakeEngine(t)
	s, cancel := startGateway(t, eng.ln.Addr().String())
	defer cancel()
	u := newUA(t, s.Addr())
	u.sendSIP(u.invite("call-4", "18")) // G.729 only
	u.recvSIP(time.Second)              // 100 Trying
	if r := u.recvSIP(time.Second); r.StatusCode != 488 {
		t.Fatalf("want 488, got %d", r.StatusCode)
	}
	u.sendSIP(u.inDialog("BYE", "nope", "x", 1))
	if r := u.recvSIP(time.Second); r.StatusCode != 481 {
		t.Fatalf("want 481, got %d", r.StatusCode)
	}
}

func TestEngineDownGives503(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := ln.Addr().String()
	ln.Close()
	s, cancel := startGateway(t, addr)
	defer cancel()
	u := newUA(t, s.Addr())
	u.sendSIP(u.invite("call-5", "0"))
	u.recvSIP(time.Second)
	if r := u.recvSIP(3 * time.Second); r.StatusCode != 503 {
		t.Fatalf("want 503, got %d", r.StatusCode)
	}
}

var testLogOut io.Writer = io.Discard // set to os.Stderr to debug
