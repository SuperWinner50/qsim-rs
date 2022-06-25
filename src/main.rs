mod state;
use state::State;

fn main() {
    let mut state = State::new(4);
    state.hadamard(0);
    state.cx(0, 1);
    state.cx(1, 2);
    state.cx(2, 3);
    println!("{:?}", state.measure_real_prob(1000));
}
