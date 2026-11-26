package g711

import "testing"

func TestUlawReference(t *testing.T) {
	in := []int16{0, 1000, -1000, 32767, -32768}
	want := []byte{0xFF, 0xCE, 0x4E, 0x80, 0x00}
	for i, s := range in {
		if got := UlawEncode(s); got != want[i] {
			t.Fatalf("UlawEncode(%d)=%#x want %#x", s, got, want[i])
		}
	}
	if UlawDecode(0x80) != 32124 || UlawDecode(0x00) != -32124 || UlawDecode(0xFF) != 0 {
		t.Fatal("UlawDecode reference mismatch")
	}
}

func TestAlawReference(t *testing.T) {
	in := []int16{0, 32767, -32768}
	want := []byte{0xD5, 0xAA, 0x2A}
	for i, s := range in {
		if got := AlawEncode(s); got != want[i] {
			t.Fatalf("AlawEncode(%d)=%#x want %#x", s, got, want[i])
		}
	}
	if AlawDecode(0xD5) != 8 || AlawDecode(0xAA) != 32256 || AlawDecode(0x2A) != -32256 {
		t.Fatal("AlawDecode reference mismatch")
	}
}

func TestMonotonicAndCodewordRoundtrip(t *testing.T) {
	for _, c := range []Codec{PCMU, PCMA} {
		prev := int16(-32768)
		for v := -32768; v <= 32767; v++ {
			d := c.Decode(c.Encode(int16(v)))
			if d < prev {
				t.Fatalf("%s not monotonic at %d", c.Name, v)
			}
			prev = d
		}
		same := 0
		for b := 0; b < 256; b++ {
			if c.Encode(c.Decode(byte(b))) == byte(b) {
				same++
			}
		}
		if same < 255 {
			t.Fatalf("%s codeword roundtrip %d/256", c.Name, same)
		}
	}
}
