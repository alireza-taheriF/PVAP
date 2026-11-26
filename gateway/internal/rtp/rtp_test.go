package rtp

import (
	"bytes"
	"testing"
)

func TestRoundtrip(t *testing.T) {
	p := &Packet{Marker: true, PayloadType: 0, Seq: 65535, Timestamp: 160, SSRC: 0xdeadbeef, Payload: []byte{1, 2, 3}}
	q, err := Unmarshal(p.Marshal())
	if err != nil || !q.Marker || q.Seq != 65535 || q.Timestamp != 160 || q.SSRC != 0xdeadbeef || !bytes.Equal(q.Payload, p.Payload) {
		t.Fatalf("%+v %v", q, err)
	}
}

func TestCSRCExtensionPadding(t *testing.T) {
	b := []byte{0xB1, 0x08, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, // V=2 P=1 X=1 CC=1, PT=8
		9, 9, 9, 9, // CSRC
		0xBE, 0xDE, 0, 1, 7, 7, 7, 7, // extension, 1 word
		0xAA, 0xBB, 0, 2} // payload AA BB + 2 padding bytes (last byte = 2)
	p, err := Unmarshal(b)
	if err != nil || p.PayloadType != 8 || !bytes.Equal(p.Payload, []byte{0xAA, 0xBB}) {
		t.Fatalf("%+v %v", p, err)
	}
	if _, err := Unmarshal([]byte{0x40, 0, 0}); err == nil {
		t.Fatal("expected error")
	}
}
