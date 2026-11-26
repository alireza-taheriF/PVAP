package sip

import (
	"strings"
	"testing"
)

const invite = "INVITE sip:agent@127.0.0.1:5060 SIP/2.0\r\n" +
	"v: SIP/2.0/UDP 10.0.0.2:5062;branch=z9hG4bK-1, SIP/2.0/UDP 10.0.0.9;branch=z9hG4bK-0\r\n" +
	"Max-Forwards: 70\r\n" +
	"f: \"Ali\" <sip:ali@10.0.0.2>;tag=abc\r\n" +
	"t: <sip:agent@127.0.0.1>\r\n" +
	"i: call-123@10.0.0.2\r\n" +
	"CSeq: 1 INVITE\r\n" +
	"m: <sip:ali@10.0.0.2:5062>\r\n" +
	"c: application/sdp\r\n" +
	"l: 133\r\n\r\n" +
	"v=0\r\no=- 1 1 IN IP4 10.0.0.2\r\ns=-\r\nc=IN IP4 10.0.0.2\r\nt=0 0\r\nm=audio 40000 RTP/AVP 8 0 101\r\na=ptime:20\r\na=rtpmap:101 telephone-event/8000\r\n"

func TestParseCompactAndMultiVia(t *testing.T) {
	m, err := Parse([]byte(invite))
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsRequest || m.Method != "INVITE" || m.RequestURI != "sip:agent@127.0.0.1:5060" {
		t.Fatalf("bad request line: %+v", m)
	}
	if got := m.All("Via"); len(got) != 2 || !strings.Contains(got[1], "z9hG4bK-0") {
		t.Fatalf("vias: %q", got)
	}
	if m.Get("Call-ID") != "call-123@10.0.0.2" || Param(m.Get("From"), "tag") != "abc" {
		t.Fatal("compact headers not canonicalized")
	}
	if URI(m.Get("Contact")) != "sip:ali@10.0.0.2:5062" {
		t.Fatalf("contact uri %q", URI(m.Get("Contact")))
	}
	if n, meth := m.CSeq(); n != 1 || meth != "INVITE" {
		t.Fatal("cseq")
	}
	if len(m.Body) != 133 {
		t.Fatalf("body len %d", len(m.Body))
	}
}

func TestResponseCopiesDialogHeaders(t *testing.T) {
	req, _ := Parse([]byte(invite))
	r := NewResponse(req, 200, "OK", "xyz")
	again, err := Parse(r.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if again.StatusCode != 200 || len(again.All("Via")) != 2 {
		t.Fatalf("response: %s", r.Bytes())
	}
	if Param(again.Get("To"), "tag") != "xyz" || again.Get("Call-ID") != req.Get("Call-ID") {
		t.Fatal("to-tag / call-id")
	}
	if again.Get("Content-Length") != "0" {
		t.Fatal("content-length")
	}
}

func TestRejectsGarbage(t *testing.T) {
	for _, s := range []string{"", "hello", "FOO bar\r\n\r\n", "SIP/2.0 abc OK\r\n\r\n",
		"INVITE sip:a SIP/2.0\r\nContent-Length: 99\r\n\r\nshort"} {
		if _, err := Parse([]byte(s)); err == nil {
			t.Fatalf("expected error for %q", s)
		}
	}
}

func TestSDPOfferAnswer(t *testing.T) {
	m, _ := Parse([]byte(invite))
	o, err := ParseSDP(m.Body)
	if err != nil {
		t.Fatal(err)
	}
	if o.ConnIP != "10.0.0.2" || o.Port != 40000 || o.Ptime != 20 {
		t.Fatalf("offer %+v", o)
	}
	pt, ok := o.ChooseCodec()
	if !ok || pt != 8 { // offerer preferred PCMA
		t.Fatalf("codec %d", pt)
	}
	a, err := ParseSDP(AnswerSDP("127.0.0.1", 20000, pt, 42))
	if err != nil || a.Port != 20000 || a.PayloadTypes[0] != 8 || a.ConnIP != "127.0.0.1" {
		t.Fatalf("answer %+v %v", a, err)
	}
}
