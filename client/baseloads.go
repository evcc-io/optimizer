package client

import "encoding/json"

// BaseLoads is the household demand of the request. The API takes either a single series or one
// series per base load, both decode into the summed series the CLI works with.
type BaseLoads []float32

func (b *BaseLoads) UnmarshalJSON(data []byte) error {
	var single []float32
	if err := json.Unmarshal(data, &single); err == nil {
		*b = single
		return nil
	}

	var loads [][]float32
	if err := json.Unmarshal(data, &loads); err != nil {
		return err
	}

	var sum []float32
	for _, load := range loads {
		for i, v := range load {
			if i == len(sum) {
				sum = append(sum, 0)
			}
			sum[i] += v
		}
	}

	*b = sum

	return nil
}
