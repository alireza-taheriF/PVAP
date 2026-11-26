// Package sip implements the small subset of SIP (RFC 3261) and SDP
// (RFC 4566) a UAS needs to answer a call: parsing and building messages,
// compact header forms, and an audio offer/answer for G.711.
package sip

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Message is a SIP request or response. Header names are stored canonical
// (long form, e.g. "Call-ID"); multi-valued headers keep their order.
type Message struct {
	IsRequest  bool
	Method     string // request
	RequestURI string // request
	StatusCode int    // response
	Reason     string // response
	Headers    []Header
	Body       []byte
}

type Header struct{ Name, Value string }

var compact = map[string]string{
	"v": "Via", "f": "From", "t": "To", "i": "Call-ID", "m": "Contact",
	"l": "Content-Length", "c": "Content-Type", "k": "Supported", "s": "Subject",
}

var canonical = map[string]string{
	"via": "Via", "from": "From", "to": "To", "call-id": "Call-ID", "cseq": "CSeq",
	"contact": "Contact", "content-length": "Content-Length", "content-type": "Content-Type",
	"max-forwards": "Max-Forwards", "user-agent": "User-Agent", "allow": "Allow",
	"supported": "Supported", "subject": "Subject", "server": "Server",
}

func canon(name string) string {
	n := strings.TrimSpace(name)
	if c, ok := compact[strings.ToLower(n)]; ok && len(n) == 1 {
		return c
	}
	if c, ok := canonical[strings.ToLower(n)]; ok {
		return c
	}
	return n
}

// Parse decodes one SIP message from a UDP datagram.
func Parse(data []byte) (*Message, error) {
	headEnd := bytes.Index(data, []byte("\r\n\r\n"))
	sepLen := 4
	if headEnd < 0 {
		headEnd = bytes.Index(data, []byte("\n\n"))
		sepLen = 2
	}
	if headEnd < 0 {
		return nil, errors.New("sip: no header terminator")
	}
	head := strings.ReplaceAll(string(data[:headEnd]), "\r\n", "\n")
	lines := strings.Split(head, "\n")
	m := &Message{}
	start := strings.TrimSpace(lines[0])
	if strings.HasPrefix(start, "SIP/2.0 ") {
		parts := strings.SplitN(start, " ", 3)
		if len(parts) < 2 {
			return nil, fmt.Errorf("sip: bad status line %q", start)
		}
		code, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, fmt.Errorf("sip: bad status code %q", parts[1])
		}
		m.StatusCode = code
		if len(parts) == 3 {
			m.Reason = parts[2]
		}
	} else {
		parts := strings.Split(start, " ")
		if len(parts) != 3 || parts[2] != "SIP/2.0" {
			return nil, fmt.Errorf("sip: bad request line %q", start)
		}
		m.IsRequest, m.Method, m.RequestURI = true, parts[0], parts[1]
	}
	for _, ln := range lines[1:] {
		if ln == "" {
			continue
		}
		if (ln[0] == ' ' || ln[0] == '\t') && len(m.Headers) > 0 { // folded line
			m.Headers[len(m.Headers)-1].Value += " " + strings.TrimSpace(ln)
			continue
		}
		name, value, ok := strings.Cut(ln, ":")
		if !ok {
			return nil, fmt.Errorf("sip: bad header %q", ln)
		}
		cn := canon(name)
		value = strings.TrimSpace(value)
		if cn == "Via" { // comma-separated Vias become separate entries
			for _, v := range splitComma(value) {
				m.Headers = append(m.Headers, Header{cn, v})
			}
			continue
		}
		m.Headers = append(m.Headers, Header{cn, value})
	}
	body := data[headEnd+sepLen:]
	if cl := m.Get("Content-Length"); cl != "" {
		n, err := strconv.Atoi(cl)
		if err != nil || n < 0 || n > len(body) {
			return nil, fmt.Errorf("sip: bad Content-Length %q", cl)
		}
		body = body[:n]
	}
	m.Body = append([]byte(nil), body...)
	return m, nil
}

func splitComma(v string) []string {
	var out []string
	depth := 0
	start := 0
	for i, r := range v {
		switch r {
		case '<':
			depth++
		case '>':
			depth--
		case ',':
			if depth == 0 {
				out = append(out, strings.TrimSpace(v[start:i]))
				start = i + 1
			}
		}
	}
	return append(out, strings.TrimSpace(v[start:]))
}

// Get returns the first value of header name ("" if absent).
func (m *Message) Get(name string) string {
	name = canon(name)
	for _, h := range m.Headers {
		if h.Name == name {
			return h.Value
		}
	}
	return ""
}

// All returns every value of header name, in order.
func (m *Message) All(name string) []string {
	name = canon(name)
	var out []string
	for _, h := range m.Headers {
		if h.Name == name {
			out = append(out, h.Value)
		}
	}
	return out
}

// Set replaces all values of name with a single value (appends if absent).
func (m *Message) Set(name, value string) {
	name = canon(name)
	kept := m.Headers[:0]
	done := false
	for _, h := range m.Headers {
		if h.Name == name {
			if !done {
				kept = append(kept, Header{name, value})
				done = true
			}
			continue
		}
		kept = append(kept, h)
	}
	m.Headers = kept
	if !done {
		m.Headers = append(m.Headers, Header{name, value})
	}
}

func (m *Message) Add(name, value string) { m.Headers = append(m.Headers, Header{canon(name), value}) }

// CSeq returns the sequence number and method.
func (m *Message) CSeq() (int, string) {
	f := strings.Fields(m.Get("CSeq"))
	if len(f) != 2 {
		return 0, ""
	}
	n, _ := strconv.Atoi(f[0])
	return n, f[1]
}

// Bytes serializes the message; Content-Length is always (re)computed.
func (m *Message) Bytes() []byte {
	var b bytes.Buffer
	if m.IsRequest {
		fmt.Fprintf(&b, "%s %s SIP/2.0\r\n", m.Method, m.RequestURI)
	} else {
		fmt.Fprintf(&b, "SIP/2.0 %d %s\r\n", m.StatusCode, m.Reason)
	}
	for _, h := range m.Headers {
		if h.Name == "Content-Length" {
			continue
		}
		fmt.Fprintf(&b, "%s: %s\r\n", h.Name, h.Value)
	}
	fmt.Fprintf(&b, "Content-Length: %d\r\n\r\n", len(m.Body))
	b.Write(m.Body)
	return b.Bytes()
}

// NewResponse builds a response to req, copying the headers RFC 3261 §8.2.6
// requires (Via, From, To, Call-ID, CSeq). toTag is added to To if missing.
func NewResponse(req *Message, code int, reason, toTag string) *Message {
	r := &Message{StatusCode: code, Reason: reason}
	for _, v := range req.All("Via") {
		r.Add("Via", v)
	}
	r.Add("From", req.Get("From"))
	to := req.Get("To")
	if toTag != "" && Param(to, "tag") == "" {
		to += ";tag=" + toTag
	}
	r.Add("To", to)
	r.Add("Call-ID", req.Get("Call-ID"))
	r.Add("CSeq", req.Get("CSeq"))
	return r
}

// Param extracts a ;name=value parameter from a header value (outside <>).
func Param(value, name string) string {
	if i := strings.LastIndex(value, ">"); i >= 0 {
		value = value[i+1:]
	}
	for _, p := range strings.Split(value, ";")[1:] {
		k, v, _ := strings.Cut(strings.TrimSpace(p), "=")
		if strings.EqualFold(k, name) {
			return v
		}
	}
	return ""
}

// URI extracts the URI from a name-addr ("Alice" <sip:a@b>;tag=1 → sip:a@b).
func URI(value string) string {
	if i := strings.Index(value, "<"); i >= 0 {
		if j := strings.Index(value[i:], ">"); j > 0 {
			return value[i+1 : i+j]
		}
	}
	u, _, _ := strings.Cut(value, ";")
	return strings.TrimSpace(u)
}
