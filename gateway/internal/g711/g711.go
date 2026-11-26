// Package g711 implements ITU-T G.711 μ-law (PCMU) and A-law (PCMA).
//
// Bit-exact with the Sun/CCITT reference (g711.c) and with pvap/audio/g711.py.
package g711

const (
	bias = 0x84
	clip = 32635
)

var ulawSegEnd = [8]int{0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF}
var alawSegEnd = [8]int{0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF}

func seg(v int, table *[8]int) int {
	for i, e := range table {
		if v <= e {
			return i
		}
	}
	return 8
}

// UlawEncode converts one 16-bit linear sample to μ-law.
func UlawEncode(s int16) byte {
	x := int(s)
	mask := 0xFF
	if x < 0 {
		x = -x
		mask = 0x7F
	}
	if x > clip {
		x = clip
	}
	x += bias
	sg := seg(x, &ulawSegEnd)
	if sg >= 8 {
		return byte(0x7F ^ mask)
	}
	return byte(((sg << 4) | ((x >> (sg + 3)) & 0x0F)) ^ mask)
}

// UlawDecode converts one μ-law byte to 16-bit linear.
func UlawDecode(b byte) int16 {
	u := int(^b)
	t := ((u & 0x0F) << 3) + bias
	t <<= (u & 0x70) >> 4
	if u&0x80 != 0 {
		return int16(bias - t)
	}
	return int16(t - bias)
}

// AlawEncode converts one 16-bit linear sample to A-law.
func AlawEncode(s int16) byte {
	x := int(s) >> 3
	mask := 0xD5
	if x < 0 {
		mask = 0x55
		x = -x - 1
	}
	sg := seg(x, &alawSegEnd)
	if sg >= 8 {
		return byte(0x7F ^ mask)
	}
	a := sg << 4
	if sg < 2 {
		a |= (x >> 1) & 0x0F
	} else {
		a |= (x >> sg) & 0x0F
	}
	return byte(a ^ mask)
}

// AlawDecode converts one A-law byte to 16-bit linear.
func AlawDecode(b byte) int16 {
	a := int(b) ^ 0x55
	t := (a & 0x0F) << 4
	sg := (a & 0x70) >> 4
	switch sg {
	case 0:
		t += 8
	case 1:
		t += 0x108
	default:
		t = (t + 0x108) << (sg - 1)
	}
	if a&0x80 != 0 {
		return int16(t)
	}
	return int16(-t)
}

// Codec is a G.711 variant identified by its static RTP payload type.
type Codec struct {
	Name        string
	PayloadType uint8
	Encode      func(int16) byte
	Decode      func(byte) int16
}

var (
	PCMU = Codec{"PCMU", 0, UlawEncode, UlawDecode}
	PCMA = Codec{"PCMA", 8, AlawEncode, AlawDecode}
)

// EncodeFrame encodes PCM samples into dst (allocated if nil or short).
func (c Codec) EncodeFrame(pcm []int16, dst []byte) []byte {
	if cap(dst) < len(pcm) {
		dst = make([]byte, len(pcm))
	}
	dst = dst[:len(pcm)]
	for i, s := range pcm {
		dst[i] = c.Encode(s)
	}
	return dst
}

// DecodeFrame decodes G.711 bytes into PCM samples.
func (c Codec) DecodeFrame(data []byte) []int16 {
	out := make([]int16, len(data))
	for i, b := range data {
		out[i] = c.Decode(b)
	}
	return out
}
