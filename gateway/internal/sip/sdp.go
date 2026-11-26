package sip

import (
	"fmt"
	"strconv"
	"strings"
)

// AudioOffer is the part of an SDP offer a G.711 UAS needs.
type AudioOffer struct {
	ConnIP       string
	Port         int
	PayloadTypes []int
	Ptime        int
}

// ParseSDP extracts the first audio media line and its connection address.
func ParseSDP(body []byte) (*AudioOffer, error) {
	o := &AudioOffer{Ptime: 20}
	sessionIP, mediaIP := "", ""
	inAudio, seenAudio := false, false
	for _, ln := range strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n") {
		ln = strings.TrimSpace(ln)
		if len(ln) < 2 || ln[1] != '=' {
			continue
		}
		k, v := ln[0], ln[2:]
		switch k {
		case 'm':
			f := strings.Fields(v)
			inAudio = len(f) >= 4 && f[0] == "audio" && !seenAudio
			if inAudio {
				seenAudio = true
				p, err := strconv.Atoi(f[1])
				if err != nil {
					return nil, fmt.Errorf("sdp: bad port %q", f[1])
				}
				o.Port = p
				for _, pt := range f[3:] {
					if n, err := strconv.Atoi(pt); err == nil {
						o.PayloadTypes = append(o.PayloadTypes, n)
					}
				}
			}
		case 'c':
			f := strings.Fields(v)
			if len(f) == 3 && f[0] == "IN" && f[1] == "IP4" {
				if !seenAudio {
					sessionIP = f[2]
				} else if inAudio {
					mediaIP = f[2]
				}
			}
		case 'a':
			if inAudio && strings.HasPrefix(v, "ptime:") {
				if n, err := strconv.Atoi(strings.TrimPrefix(v, "ptime:")); err == nil {
					o.Ptime = n
				}
			}
		}
	}
	if !seenAudio {
		return nil, fmt.Errorf("sdp: no audio media")
	}
	o.ConnIP = mediaIP
	if o.ConnIP == "" {
		o.ConnIP = sessionIP
	}
	if o.ConnIP == "" {
		return nil, fmt.Errorf("sdp: no connection address")
	}
	return o, nil
}

// ChooseCodec picks PCMU (0) or PCMA (8) in the offerer's preference order.
func (o *AudioOffer) ChooseCodec() (int, bool) {
	for _, pt := range o.PayloadTypes {
		if pt == 0 || pt == 8 {
			return pt, true
		}
	}
	return -1, false
}

// AnswerSDP builds the answer for one G.711 payload type.
func AnswerSDP(ip string, port, pt int, sessID int64) []byte {
	name := "PCMU"
	if pt == 8 {
		name = "PCMA"
	}
	return []byte(fmt.Sprintf(
		"v=0\r\no=pvap %d %d IN IP4 %s\r\ns=PVAP\r\nc=IN IP4 %s\r\nt=0 0\r\n"+
			"m=audio %d RTP/AVP %d\r\na=rtpmap:%d %s/8000\r\na=ptime:20\r\na=sendrecv\r\n",
		sessID, sessID, ip, ip, port, pt, pt, name))
}
