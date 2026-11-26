// Package gw is the PVAP telephony gateway: a minimal SIP UAS over UDP that
// answers calls with G.711, terminates RTP, and bridges 20 ms PCM frames to
// the Python dialog engine over the framed TCP protocol in package bridge.
//
// Scope (MVP): direct SIP from a trunk/PBX or softphone on a trusted network
// (no auth, no TLS/SRTP, no REGISTER), UDP only, PCMU/PCMA at 20 ms,
// symmetric RTP latching for NAT, in-order playout without a jitter buffer.
package gw

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"github.com/alireza-taheriF/pvap/gateway/internal/bridge"
	"github.com/alireza-taheriF/pvap/gateway/internal/g711"
	"github.com/alireza-taheriF/pvap/gateway/internal/rtp"
	"github.com/alireza-taheriF/pvap/gateway/internal/sip"
)

const (
	frameSamples = 160 // 20 ms @ 8 kHz
	frameDur     = 20 * time.Millisecond
	playoutCap   = 750 // 15 s of queued agent audio
)

type Config struct {
	SIPAddr    string // e.g. "0.0.0.0:5060"
	PublicIP   string // address advertised in SDP / Contact
	RTPPortMin int
	RTPPortMax int
	EngineAddr string // dialog engine TCP address
	Logger     *log.Logger
}

type Server struct {
	cfg   Config
	conn  *net.UDPConn
	log   *log.Logger
	mu    sync.Mutex
	calls map[string]*call
	port  int
}

func New(cfg Config) (*Server, error) {
	if cfg.Logger == nil {
		cfg.Logger = log.Default()
	}
	if cfg.RTPPortMin == 0 {
		cfg.RTPPortMin, cfg.RTPPortMax = 20000, 20999
	}
	addr, err := net.ResolveUDPAddr("udp4", cfg.SIPAddr)
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenUDP("udp4", addr)
	if err != nil {
		return nil, err
	}
	if cfg.PublicIP == "" {
		cfg.PublicIP = conn.LocalAddr().(*net.UDPAddr).IP.String()
		if cfg.PublicIP == "0.0.0.0" {
			cfg.PublicIP = "127.0.0.1"
		}
	}
	return &Server{cfg: cfg, conn: conn, log: cfg.Logger, calls: map[string]*call{}, port: cfg.RTPPortMin}, nil
}

func (s *Server) Addr() *net.UDPAddr { return s.conn.LocalAddr().(*net.UDPAddr) }

// ActiveCalls reports the number of live calls.
func (s *Server) ActiveCalls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.calls)
}

// Serve handles SIP until ctx is cancelled.
func (s *Server) Serve(ctx context.Context) error {
	go func() {
		<-ctx.Done()
		s.mu.Lock()
		calls := make([]*call, 0, len(s.calls))
		for _, c := range s.calls {
			calls = append(calls, c)
		}
		s.mu.Unlock()
		for _, c := range calls {
			c.end(true, true)
		}
		s.conn.Close()
	}()
	buf := make([]byte, 65535)
	for {
		n, from, err := s.conn.ReadFromUDP(buf)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		msg, err := sip.Parse(buf[:n])
		if err != nil {
			s.log.Printf("sip: drop malformed datagram from %s: %v", from, err)
			continue
		}
		if msg.IsRequest {
			s.handleRequest(msg, from)
		}
	}
}

func (s *Server) send(m *sip.Message, to *net.UDPAddr) {
	if _, err := s.conn.WriteToUDP(m.Bytes(), to); err != nil {
		s.log.Printf("sip: send to %s: %v", to, err)
	}
}

func (s *Server) reply(req *sip.Message, to *net.UDPAddr, code int, reason, tag string) {
	r := sip.NewResponse(req, code, reason, tag)
	r.Add("Server", "pvap-gateway")
	s.send(r, to)
}

func (s *Server) lookup(id string) *call {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[id]
}

func (s *Server) handleRequest(req *sip.Message, from *net.UDPAddr) {
	id := req.Get("Call-ID")
	switch req.Method {
	case "OPTIONS":
		r := sip.NewResponse(req, 200, "OK", randHex(4))
		r.Add("Allow", "INVITE, ACK, BYE, CANCEL, OPTIONS")
		s.send(r, from)
	case "INVITE":
		if c := s.lookup(id); c != nil { // retransmission or re-INVITE
			s.conn.WriteToUDP(c.ok200, from)
			return
		}
		s.newCall(req, from)
	case "ACK":
		if c := s.lookup(id); c != nil {
			c.ack()
		}
	case "BYE":
		c := s.lookup(id)
		if c == nil {
			s.reply(req, from, 481, "Call/Transaction Does Not Exist", "")
			return
		}
		s.reply(req, from, 200, "OK", c.toTag)
		c.end(false, true)
	case "CANCEL":
		c := s.lookup(id)
		if c == nil {
			s.reply(req, from, 481, "Call/Transaction Does Not Exist", "")
			return
		}
		s.reply(req, from, 200, "OK", c.toTag)
		if !c.isAcked() {
			r := sip.NewResponse(c.invite, 487, "Request Terminated", c.toTag)
			s.send(r, from)
		}
		c.end(false, true)
	default:
		s.reply(req, from, 501, "Not Implemented", "")
	}
}

func (s *Server) allocRTP() (*net.UDPConn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	span := s.cfg.RTPPortMax - s.cfg.RTPPortMin + 1
	for i := 0; i < span; i++ {
		p := s.port
		s.port++
		if s.port > s.cfg.RTPPortMax {
			s.port = s.cfg.RTPPortMin
		}
		if p%2 == 1 { // RTP on even ports by convention
			continue
		}
		conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4zero, Port: p})
		if err == nil {
			return conn, nil
		}
	}
	return nil, errors.New("no free RTP port")
}

func (s *Server) newCall(req *sip.Message, from *net.UDPAddr) {
	tag := randHex(6)
	s.reply(req, from, 100, "Trying", "")

	offer, err := sip.ParseSDP(req.Body)
	if err != nil {
		s.reply(req, from, 488, "Not Acceptable Here", tag)
		return
	}
	pt, ok := offer.ChooseCodec()
	if !ok {
		s.reply(req, from, 488, "Not Acceptable Here", tag)
		return
	}
	codec := g711.PCMU
	if pt == 8 {
		codec = g711.PCMA
	}
	remoteRTP, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("%s:%d", offer.ConnIP, offer.Port))
	if err != nil {
		s.reply(req, from, 488, "Not Acceptable Here", tag)
		return
	}
	rtpConn, err := s.allocRTP()
	if err != nil {
		s.reply(req, from, 503, "Service Unavailable", tag)
		return
	}
	engine, err := net.DialTimeout("tcp", s.cfg.EngineAddr, 2*time.Second)
	if err != nil {
		rtpConn.Close()
		s.log.Printf("call %s: engine unreachable: %v", req.Get("Call-ID"), err)
		s.reply(req, from, 503, "Service Unavailable", tag)
		return
	}

	c := &call{
		srv: s, id: req.Get("Call-ID"), toTag: tag, invite: req,
		remoteSIP: from, remoteRTP: remoteRTP, codec: codec,
		rtpConn: rtpConn, engine: engine,
		playout: make(chan []int16, playoutCap), clear: make(chan struct{}, 1),
		acked: make(chan struct{}), done: make(chan struct{}),
		ssrc: randU32(),
	}
	meta, _ := json.Marshal(map[string]any{
		"call_id": c.id, "from": sip.URI(req.Get("From")), "to": sip.URI(req.Get("To")),
		"codec": codec.Name, "sample_rate": 8000,
	})
	if err := c.engineWrite(bridge.Start, meta); err != nil {
		c.closeConns()
		s.reply(req, from, 503, "Service Unavailable", tag)
		return
	}

	localPort := rtpConn.LocalAddr().(*net.UDPAddr).Port
	ok200 := sip.NewResponse(req, 200, "OK", tag)
	ok200.Add("Contact", fmt.Sprintf("<sip:pvap@%s:%d>", s.cfg.PublicIP, s.Addr().Port))
	ok200.Add("Content-Type", "application/sdp")
	ok200.Add("Server", "pvap-gateway")
	ok200.Body = sip.AnswerSDP(s.cfg.PublicIP, localPort, pt, time.Now().Unix())
	c.ok200 = ok200.Bytes()
	c.localTo = ok200.Get("To")

	s.mu.Lock()
	s.calls[c.id] = c
	s.mu.Unlock()
	s.log.Printf("call %s: answered %s, rtp %d ↔ %s", c.id, codec.Name, localPort, remoteRTP)

	go c.retransmit200()
	go c.rtpIn()
	go c.engineIn()
	go c.rtpOut()
}

type call struct {
	srv       *Server
	id, toTag string
	localTo   string
	invite    *sip.Message
	remoteSIP *net.UDPAddr
	codec     g711.Codec
	rtpConn   *net.UDPConn
	engine    net.Conn
	ok200     []byte
	ssrc      uint32

	rmu       sync.Mutex
	remoteRTP *net.UDPAddr
	latched   bool

	wmu     sync.Mutex // engine writes
	playout chan []int16
	clear   chan struct{}

	ackOnce sync.Once
	acked   chan struct{}
	endOnce sync.Once
	done    chan struct{}
}

func (c *call) ack() { c.ackOnce.Do(func() { close(c.acked) }) }
func (c *call) isAcked() bool {
	select {
	case <-c.acked:
		return true
	default:
		return false
	}
}

func (c *call) engineWrite(kind byte, payload []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	c.engine.SetWriteDeadline(time.Now().Add(2 * time.Second))
	return bridge.Write(c.engine, kind, payload)
}

// retransmit200 repeats the 2xx until ACK (RFC 3261 §13.3.1.4: T1 doubling to T2, 64·T1 total).
func (c *call) retransmit200() {
	interval, total := 500*time.Millisecond, time.Duration(0)
	for total < 32*time.Second {
		c.srv.conn.WriteToUDP(c.ok200, c.remoteSIP)
		select {
		case <-c.acked:
			return
		case <-c.done:
			return
		case <-time.After(interval):
		}
		total += interval
		if interval *= 2; interval > 4*time.Second {
			interval = 4 * time.Second
		}
	}
	c.srv.log.Printf("call %s: no ACK, hanging up", c.id)
	c.end(true, true)
}

func (c *call) rtpIn() {
	buf := make([]byte, 2048)
	for {
		n, src, err := c.rtpConn.ReadFromUDP(buf)
		if err != nil {
			return
		}
		p, err := rtp.Unmarshal(buf[:n])
		if err != nil || p.PayloadType != c.codec.PayloadType {
			continue // ignore DTMF events, comfort noise, junk
		}
		c.rmu.Lock()
		if !c.latched { // symmetric RTP: answer to where media really comes from
			c.remoteRTP, c.latched = src, true
		}
		c.rmu.Unlock()
		if err := c.engineWrite(bridge.Audio, bridge.PCMToBytes(c.codec.DecodeFrame(p.Payload))); err != nil {
			c.end(true, false)
			return
		}
	}
}

func (c *call) engineIn() {
	for {
		kind, payload, err := bridge.Read(c.engine)
		if err != nil {
			c.end(true, false)
			return
		}
		switch kind {
		case bridge.Audio:
			frame := bridge.BytesToPCM(payload)
			select {
			case c.playout <- frame:
			default: // queue full: drop oldest, keep newest
				select {
				case <-c.playout:
				default:
				}
				c.playout <- frame
			}
		case bridge.Clear:
			select {
			case c.clear <- struct{}{}:
			default:
			}
		case bridge.Hangup:
			c.end(true, false)
			return
		}
	}
}

func (c *call) drain() int {
	n := 0
	for {
		select {
		case <-c.playout:
			n++
		default:
			return n
		}
	}
}

func (c *call) rtpOut() {
	t := time.NewTicker(frameDur)
	defer t.Stop()
	silence := make([]int16, frameSamples)
	pkt := rtp.Packet{PayloadType: c.codec.PayloadType, SSRC: c.ssrc, Seq: uint16(randU32()), Timestamp: randU32()}
	enc := make([]byte, frameSamples)
	talking := false
	for {
		select {
		case <-c.done:
			return
		case <-c.clear:
			c.drain()
			talking = false
			continue
		case <-t.C:
		}
		frame := silence
		select {
		case f := <-c.playout:
			frame, pkt.Marker = f, !talking
			talking = true
		default:
			pkt.Marker, talking = false, false
		}
		pkt.Payload = c.codec.EncodeFrame(frame, enc)
		c.rmu.Lock()
		dst := c.remoteRTP
		c.rmu.Unlock()
		c.rtpConn.WriteToUDP(pkt.Marshal(), dst)
		pkt.Seq++
		pkt.Timestamp += frameSamples
	}
}

// end tears the call down once. sendBye: tell the SIP peer; notifyEngine: tell Python.
func (c *call) end(sendBye, notifyEngine bool) {
	c.endOnce.Do(func() {
		close(c.done)
		if notifyEngine {
			c.engineWrite(bridge.Hangup, nil)
		}
		c.closeConns()
		forget := func() {
			c.srv.mu.Lock()
			delete(c.srv.calls, c.id)
			c.srv.mu.Unlock()
			c.srv.log.Printf("call %s: ended", c.id)
		}
		if !sendBye {
			forget()
			return
		}
		// A UAS may only send BYE once the dialog is confirmed (ACK). If the
		// ACK is still in flight, keep the call findable and wait briefly.
		go func() {
			defer forget()
			select {
			case <-c.acked:
				c.srv.send(c.bye(), c.remoteSIP)
			case <-time.After(2 * time.Second):
			}
		}()
	})
}

func (c *call) closeConns() {
	c.rtpConn.Close()
	c.engine.Close()
}

func (c *call) bye() *sip.Message {
	target := sip.URI(c.invite.Get("Contact"))
	if target == "" {
		target = sip.URI(c.invite.Get("From"))
	}
	m := &sip.Message{IsRequest: true, Method: "BYE", RequestURI: target}
	m.Add("Via", fmt.Sprintf("SIP/2.0/UDP %s:%d;branch=z9hG4bK%s;rport", c.srv.cfg.PublicIP, c.srv.Addr().Port, randHex(8)))
	m.Add("Max-Forwards", "70")
	m.Add("From", c.localTo)
	m.Add("To", c.invite.Get("From"))
	m.Add("Call-ID", c.id)
	m.Add("CSeq", "1 BYE")
	m.Add("User-Agent", "pvap-gateway")
	return m
}

func randHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func randU32() uint32 {
	b := make([]byte, 4)
	rand.Read(b)
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}
