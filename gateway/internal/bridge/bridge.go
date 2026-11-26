// Package bridge speaks the gateway ↔ dialog-engine framing protocol
// (see pvap/bridge/protocol.py): type(1) | length(4, BE) | payload.
package bridge

import (
	"encoding/binary"
	"fmt"
	"io"
)

const (
	Start  byte = 0x01
	Audio  byte = 0x02
	Clear  byte = 0x03
	Hangup byte = 0x04

	MaxPayload = 64 * 1024
)

func Write(w io.Writer, kind byte, payload []byte) error {
	if len(payload) > MaxPayload {
		return fmt.Errorf("bridge: payload too large (%d)", len(payload))
	}
	hdr := [5]byte{kind}
	binary.BigEndian.PutUint32(hdr[1:], uint32(len(payload)))
	if _, err := w.Write(append(hdr[:], payload...)); err != nil {
		return err
	}
	return nil
}

func Read(r io.Reader) (byte, []byte, error) {
	var hdr [5]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return 0, nil, err
	}
	n := binary.BigEndian.Uint32(hdr[1:])
	if n > MaxPayload {
		return 0, nil, fmt.Errorf("bridge: payload too large (%d)", n)
	}
	p := make([]byte, n)
	if _, err := io.ReadFull(r, p); err != nil {
		return 0, nil, err
	}
	return hdr[0], p, nil
}

// PCMToBytes / BytesToPCM convert int16 little-endian audio payloads.
func PCMToBytes(pcm []int16) []byte {
	b := make([]byte, 2*len(pcm))
	for i, s := range pcm {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

func BytesToPCM(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}
