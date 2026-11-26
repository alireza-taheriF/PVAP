// Command pvap-gateway answers SIP calls and bridges their audio to the PVAP
// dialog engine.
//
//	pvap-gateway -sip 0.0.0.0:5060 -public-ip 192.168.1.10 -engine 127.0.0.1:9300
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/alireza-taheriF/pvap/gateway/internal/gw"
)

func main() {
	sipAddr := flag.String("sip", "0.0.0.0:5060", "SIP UDP listen address")
	publicIP := flag.String("public-ip", "", "IP advertised in SDP/Contact (default: listen IP, or 127.0.0.1)")
	engine := flag.String("engine", "127.0.0.1:9300", "dialog engine TCP address")
	rtpMin := flag.Int("rtp-min", 20000, "lowest RTP port")
	rtpMax := flag.Int("rtp-max", 20999, "highest RTP port")
	flag.Parse()

	logger := log.New(os.Stderr, "pvap-gateway ", log.LstdFlags|log.Lmicroseconds)
	s, err := gw.New(gw.Config{SIPAddr: *sipAddr, PublicIP: *publicIP, EngineAddr: *engine,
		RTPPortMin: *rtpMin, RTPPortMax: *rtpMax, Logger: logger})
	if err != nil {
		logger.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	logger.Printf("SIP on %s, engine %s", s.Addr(), *engine)
	if err := s.Serve(ctx); err != nil {
		logger.Fatal(err)
	}
}
