#![allow(dead_code)]

use ndarray::{
    linalg::{general_mat_vec_mul, kron},
    Array1, Array2,
};
use rand::prelude::Distribution;
use std::collections::BTreeMap;

/// A quantum statevector holding `size` qubits
#[derive(Clone, Debug)]
pub struct State {
    size: usize,
    state: Array1<f64>,
    identities: Vec<Array2<f64>>,
}

impl State {
    fn zeros(&self) -> Array2<f64> {
        Array2::zeros((1 << self.size, 1 << self.size))
    }

    /// Creates a State with `size` qubits
    pub fn new(size: usize) -> Self {
        let mut state: Array1<f64> = Array1::zeros(1 << size);
        state[0] = 1.0;

        // Create identities so I dont have to re-compute every time
        let mut identities = vec![Array2::ones((1, 1))];
        for i in 1..size {
            identities.push(Array2::eye(1 << i));
        }

        Self {
            size,
            state,
            identities,
        }
    }

    /// Compute a single gate op with the given 2x2 matrix and 0-based index
    pub fn single_gate(&mut self, op: Array2<f64>, x: usize) {
        assert!(op.dim() == (2, 2));

        let size1 = self.size - x - 1;
        let mut state = self.identities[size1].clone();
        state = kron(&state, &op);

        let state2 = &self.identities[x];
        state = kron(&state, state2);

        // Multiply state with self.state, while also storing in self.state
        general_mat_vec_mul(1.0, &state, &self.state.clone(), 0.0, &mut self.state);
    }

    /// Hadamard gate
    pub fn hadamard(&mut self, x0: usize) {
        let op = (2.0f64.sqrt() / 2.0)
            * Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, -1.0]).unwrap();

        self.single_gate(op, x0);
    }

    /// X / NOT gate
    pub fn x(&mut self, x: usize) {
        let op = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        self.single_gate(op, x)
    }

    /// Z gate
    pub fn z(&mut self, x: usize) {
        let op = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        self.single_gate(op, x);
    }

    /// CX / CNOT gate with a given 0-based control and target qubit
    pub fn cx(&mut self, control: usize, target: usize) {
        let braket0 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let braket1 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let x1 = target;

        // If control qubit is greater than target run first option, otherwise reverse it
        match control {
            x0 if x0 > x1 => {
                let mut state_0 = kron(&self.identities[x1], &braket0);
                state_0 = kron(&state_0, &self.identities[self.size - x1 - 1]);

                let mut state_1 = kron(&self.identities[x1], &braket1);
                state_1 = kron(&state_1, &self.identities[x0 - x1 - 1]);
                state_1 = kron(&state_1, &x);
                state_1 = kron(&state_1, &self.identities[self.size - x0 - 1]);

                let state = state_0 + state_1;
                general_mat_vec_mul(1.0, &state, &self.state.clone(), 0.0, &mut self.state);
            }
            x0 if x0 < x1 => {
                let mut state_0 = kron(&braket0, &self.identities[x0]);
                state_0 = kron(&self.identities[self.size - x0 - 1], &state_0);

                let mut state_1 = kron(&braket1, &self.identities[x0]);
                state_1 = kron(&self.identities[x1 - x0 - 1], &state_1);
                state_1 = kron(&x, &state_1);
                state_1 = kron(&self.identities[self.size - x1 - 1], &state_1);

                let state = state_0 + state_1;
                general_mat_vec_mul(1.0, &state, &self.state.clone(), 0.0, &mut self.state);
            }
            _ => panic!("Cannot use same control and target qubit with CX gate"),
        }
    }

    /// Measures the theoretical probability
    pub fn measure_prob(&self) -> BTreeMap<String, f64> {
        // Generate all numbers up to 2^self.size and convert to binary
        // Since it uses u64s this means max would be 63 qubits, but that shouldnt be a problem
        (0..1u64 << self.size)
            .zip(self.state.iter())
            .map(|(bits, val)| (format!("{:0>size$b}", bits, size = self.size), val.powi(2)))
            .filter(|value| value.1 != 0.0)
            .collect()
    }

    /// Measures the results for a given amount of shots
    pub fn measure_real(&self, shots: usize) -> BTreeMap<String, usize> {
        let probs = self.measure_prob();
        let rng_uni = rand::distributions::WeightedIndex::new(probs.values()).unwrap();
        let mut results: BTreeMap<String, usize> = probs
            .keys()
            .cloned()
            .zip(std::iter::once(0).cycle())
            .collect();

        for index in rng_uni.sample_iter(rand::thread_rng()).take(shots) {
            *results.iter_mut().nth(index).unwrap().1 += 1
        }

        results
    }

    /// Measures a probability for a given amount of shots
    pub fn measure_real_prob(&self, shots: usize) -> BTreeMap<String, f64> {
        let real = self.measure_real(shots);
        real.into_iter()
            .map(|(bits, v)| (bits, v as f64 / shots as f64))
            .collect()
    }
}
