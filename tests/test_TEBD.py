import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from seemps.state import CanonicalMPS, DEFAULT_STRATEGY, product_state, NO_TRUNCATION
from seemps.tools import *
from .tools import *
from seemps.evolution import *
from seemps.hamiltonians import HeisenbergHamiltonian


def random_wavefunction(n):
    ψ = np.random.rand(n) - 0.5
    return ψ / np.linalg.norm(ψ)


class EvolutionTestCase(TestCase):
    Heisenberg2 = 0.25 * (np.kron(σx, σx) + np.kron(σy, σy) + np.kron(σz, σz))

    def random_initial_state(self, size: int) -> MPS:
        states = np.random.randn(size, 2) + 1j * np.random.randn(size, 2)
        for n in range(size):
            states[n, :] /= np.linalg.norm(states[n, :])
        return product_state(states)


class TestPairwiseUnitaries(EvolutionTestCase):
    def test_pairwise_unitaries_matrices(self):
        """Check that the nearest-neighbor unitary matrices are built properly."""
        dt = 0.33
        H = HeisenbergHamiltonian(4)
        pairwiseU = PairwiseUnitaries(H, dt, DEFAULT_STRATEGY)
        exactU = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        self.assertSimilar(pairwiseU.U[0], exactU)
        self.assertSimilar(pairwiseU.U[1], exactU)
        self.assertSimilar(pairwiseU.U[2], exactU)

    def test_pairwise_unitaries_two_sites(self):
        """Verify the exact action of the PairwiseUnitaries on two sites."""
        dt = 0.33
        H = HeisenbergHamiltonian(2)
        exactU = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        pairwiseU = PairwiseUnitaries(H, dt, DEFAULT_STRATEGY)
        mps = self.random_initial_state(2)
        self.assertSimilar(
            pairwiseU.U[0].reshape(4, 4) @ mps.to_vector(), exactU @ mps.to_vector()
        )
        self.assertSimilar(pairwiseU.apply(mps).to_vector(), exactU @ mps.to_vector())

    def test_pairwise_unitaries_three_sites(self):
        """Verify the exact action of the PairwiseUnitaries on three sites."""
        dt = 0.33
        H = HeisenbergHamiltonian(3)
        exactU12 = np.kron(scipy.linalg.expm(-1j * dt * self.Heisenberg2), np.eye(2))
        exactU23 = np.kron(np.eye(2), scipy.linalg.expm(-1j * dt * self.Heisenberg2))
        pairwiseU = PairwiseUnitaries(H, dt, DEFAULT_STRATEGY)
        mps = self.random_initial_state(3)
        #
        # When center = 0, unitaries are applied left to right
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=0)).to_vector(),
            exactU23 @ exactU12 @ mps.to_vector(),
        )
        #
        # Otherwise, they are applied right to left
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=2)).to_vector(),
            exactU12 @ exactU23 @ mps.to_vector(),
        )

    def test_pairwise_unitaries_four_sites(self):
        """Verify the exact action of the PairwiseUnitaries on four sites."""
        dt = 0.33
        H = HeisenbergHamiltonian(4)
        exactU12 = np.kron(scipy.linalg.expm(-1j * dt * self.Heisenberg2), np.eye(4))
        exactU23 = np.kron(
            np.eye(2),
            np.kron(scipy.linalg.expm(-1j * dt * self.Heisenberg2), np.eye(2)),
        )
        exactU34 = np.kron(np.eye(4), scipy.linalg.expm(-1j * dt * self.Heisenberg2))
        pairwiseU = PairwiseUnitaries(H, dt, NO_TRUNCATION)
        mps = self.random_initial_state(4)
        #
        # When center = 0, unitaries are applied left to right
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=0)).to_vector(),
            exactU34 @ exactU23 @ exactU12 @ mps.to_vector(),
        )
        #
        # Otherwise, they are applied right to left
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=2)).to_vector(),
            exactU12 @ exactU23 @ exactU34 @ mps.to_vector(),
        )

    def test_pairwise_unitaries_center_strategy(self):
        """Verify how PairwiseUnitaries decides the order of application."""
        dt = 0.33
        N = 7
        H = HeisenbergHamiltonian(N)
        pairwiseU = PairwiseUnitaries(H, dt, NO_TRUNCATION)
        mps = self.random_initial_state(N)
        mps_from_left = pairwiseU.apply(CanonicalMPS(mps, center=0))
        mps_from_right = pairwiseU.apply(CanonicalMPS(mps, center=N - 1))
        self.assertSimilar(mps_from_left, pairwiseU.apply(CanonicalMPS(mps, center=0)))
        self.assertSimilar(mps_from_left, pairwiseU.apply(CanonicalMPS(mps, center=1)))
        self.assertSimilar(mps_from_left, pairwiseU.apply(CanonicalMPS(mps, center=2)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=3)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=4)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=5)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=6)))


class TestTrotter2nd(EvolutionTestCase):
    def test_trotter_2nd_order_two_sites(self):
        dt = 0.33
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(2), dt)
        U12 = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        mps = self.random_initial_state(2)
        self.assertSimilar(trotterU.apply(mps).to_vector(), U12 @ mps.to_vector())

    def test_trotter_2nd_order_three_sites(self):
        dt = 0.33
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(3), dt)
        U2 = scipy.linalg.expm(-0.5j * dt * self.Heisenberg2)
        U23 = np.kron(np.eye(2), U2)
        U12 = np.kron(U2, np.eye(2))
        mps = self.random_initial_state(3)
        self.assertSimilar(
            trotterU.apply(mps).to_vector(),
            U12 @ (U23 @ (U23 @ (U12 @ mps.to_vector()))),
        )

    def test_trotter_2nd_order_four_sites(self):
        dt = 0.33
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(4), dt)
        U2 = scipy.linalg.expm(-0.5j * dt * self.Heisenberg2)
        U34 = np.kron(np.eye(4), U2)
        U23 = np.kron(np.eye(2), np.kron(U2, np.eye(2)))
        U12 = np.kron(U2, np.eye(4))
        mps = self.random_initial_state(4)
        self.assertSimilar(
            trotterU.apply(mps).to_vector(),
            U12 @ (U23 @ (U34 @ (U34 @ (U23 @ (U12 @ mps.to_vector()))))),
        )


class TestTrotter3rd(EvolutionTestCase):
    def test_trotter_3rd_order_two_sites(self):
        dt = 0.33
        trotterU = Trotter3rdOrder(HeisenbergHamiltonian(2), dt)
        U12 = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        mps = self.random_initial_state(2)
        self.assertSimilar(trotterU.apply(mps).to_vector(), U12 @ mps.to_vector())

    def test_trotter_3rd_order_three_sites(self):
        dt = 0.33
        trotterU = Trotter3rdOrder(HeisenbergHamiltonian(3), dt)
        U2half = scipy.linalg.expm(-0.5j * dt * self.Heisenberg2)
        U2 = scipy.linalg.expm(-0.25j * dt * self.Heisenberg2)
        U23 = np.kron(np.eye(2), U2)
        U12 = np.kron(U2, np.eye(2))
        U23half = np.kron(np.eye(2), U2half)
        U12half = np.kron(U2half, np.eye(2))
        mps = self.random_initial_state(3)
        self.assertSimilar(
            trotterU.apply(mps).to_vector(),
            U23 @ (U12 @ (U12half @ (U23half @ (U23 @ (U12 @ mps.to_vector()))))),
        )
