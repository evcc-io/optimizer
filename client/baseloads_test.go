package client

import (
	"encoding/json"
	"testing"
)

func TestBaseLoadsUnmarshal(t *testing.T) {
	for _, tc := range []string{`[400,600]`, `[[400,600]]`, `[[100,200],[300,400]]`} {
		var loads BaseLoads
		if err := json.Unmarshal([]byte(tc), &loads); err != nil {
			t.Fatalf("%s: %v", tc, err)
		}
		if len(loads) != 2 || loads[0] != 400 || loads[1] != 600 {
			t.Errorf("%s: got %v", tc, loads)
		}
	}
}
