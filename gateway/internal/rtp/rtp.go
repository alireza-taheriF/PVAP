// Package rtp marshals RTP (RFC 3550) packets without extensions or CSRCs
// beyond what a G.711 voice stream needs.
package rtp

import (
	"encoding/binary"
	"errors"
)

type Packet struct {
	Marker      bool
	PayloadType uint8
	Seq         uint16
	Timestamp   uint32
	SSRC        uint32
	Payload     []byte
}

// Marshal writes a version-2 RTP packet.
func (p *Packet) Marshal() []byte {
	b := make([]byte, 12+len(p.Payload))
	b[0] = 0x80
	b[1] = p.PayloadType & 0x7F
	if p.Marker {
		b[1] |= 0x80
	}
	binary.BigEndian.PutUint16(b[2:], p.Seq)
	binary.BigEndian.PutUint32(b[4:], p.Timestamp)
	binary.BigEndian.PutUint32(b[8:], p.SSRC)
	copy(b[12:], p.Payload)
	return b
}

// Unmarshal parses an RTP packet, skipping CSRCs, header extensions and padding.
func Unmarshal(b []byte) (*Packet, error) {
	if len(b) < 12 || b[0]>>6 != 2 {
		return nil, errors.New("rtp: not an RTP v2 packet")
	}
	cc := int(b[0] & 0x0F)
	off := 12 + 4*cc
	if b[0]&0x10 != 0 { // extension
		if len(b) < off+4 {
			return nil, errors.New("rtp: short extension")
		}
		off += 4 + 4*int(binary.BigEndian.Uint16(b[off+2:]))
	}
	end := len(b)
	if b[0]&0x20 != 0 { // padding
		end -= int(b[len(b)-1])
	}
	if off > end {
		return nil, errors.New("rtp: truncated")
	}
	return &Packet{
		Marker:      b[1]&0x80 != 0,
		PayloadType: b[1] & 0x7F,
		Seq:         binary.BigEndian.Uint16(b[2:]),
		Timestamp:   binary.BigEndian.Uint32(b[4:]),
		SSRC:        binary.BigEndian.Uint32(b[8:]),
		Payload:     append([]byte(nil), b[off:end]...),
	}, nil
}
