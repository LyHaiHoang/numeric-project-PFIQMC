program ising_magnetization_vs_T
    implicit none
    integer, parameter :: L = 100, n_mcs = 5000
    integer, parameter :: N = L * L
    real(8), parameter :: kb = 1.0d0, Jc = 1.0d0, H = 0.0d0
    real(8), dimension(L,L) :: spins
    real(8), dimension(:), allocatable :: T_vals, mT_vals
    real(8) :: T, T_start, T_end, dT
    integer :: iT, step
    character(len=100) :: filename

    call random_seed()

    ! === Temperature scan parameters ===
    T_start = 1.0d0
    T_end   = 4.0d0
    dT      = 0.1d0
    allocate(T_vals(int((T_end - T_start)/dT) + 1))
    allocate(mT_vals(size(T_vals)))

    ! === Fill temperature vector ===
    do iT = 1, size(T_vals)
        T_vals(iT) = T_start + dT * (iT - 1)
    end do

    ! === Initialize spins ===
    call spin_init(spins, L, 1)

    print *, "=== Simulation M(T) for 2D Ising Model ==="

    ! === Loop over temperature ===
    do iT = 1, size(T_vals)
        T = T_vals(iT)
        print *, "Temperature: ", T

        ! Thermalization
        do step = 1, n_mcs
            call metropolis(spins, L, H, Jc, kb, T)
        end do

        ! Compute mean magnetization
        mT_vals(iT) = magnetization(spins, L)
    end do

    ! === Save results ===
    filename = "mT_L" // trim(adjustl(itoa(L))) // "_step" // trim(adjustl(itoa(n_mcs))) // "_H0.dat"
    open(unit=10, file=filename, status="replace")
    write(10,'(A)') "# T(K)   M(T)"
    do iT = 1, size(T_vals)
        write(10,'(2F12.6)') T_vals(iT), mT_vals(iT)
    end do
    close(10)
    print *, "Results saved in ", trim(filename)

contains

    !====================================================
    subroutine spin_init(spins, L, mode)
        integer, intent(in) :: L, mode
        real(8), intent(out) :: spins(L,L)
        if (mode == 1) then
            spins = 1.0d0
        else
            spins = -1.0d0
        end if
    end subroutine spin_init
    !====================================================

    real(8) function delta_E(spins, L, ii, jj, H, J)
        integer, intent(in) :: L, ii, jj
        real(8), intent(in) :: spins(L,L), H, J
        integer :: ip, im, jp, jm
        real(8) :: s, nn
        ip = mod(ii, L) + 1
        im = mod(ii - 2, L) + 1
        jp = mod(jj, L) + 1
        jm = mod(jj - 2, L) + 1
        s  = spins(ii, jj)
        nn = spins(ip, jj) + spins(im, jj) + spins(ii, jp) + spins(ii, jm)
        delta_E = 2.0d0 * s * (J * nn + H)
    end function delta_E
    !====================================================

    subroutine metropolis(spins, L, H, J, kb, T)
        integer, intent(in) :: L
        real(8), intent(inout) :: spins(L,L)
        real(8), intent(in) :: H, J, kb, T
        real(8) :: dE, beta, r
        integer :: k, ii, jj
        beta = 1.0d0 / (kb * T)

        do k = 1, L * L
            call random_number(r)
            ii = 1 + int(r * L)
            call random_number(r)
            jj = 1 + int(r * L)
            dE = delta_E(spins, L, ii, jj, H, J)
            call random_number(r)
            if (dE <= 0.0d0 .or. r < exp(-beta * dE)) then
                spins(ii, jj) = -spins(ii, jj)
            end if
        end do
    end subroutine metropolis
    !====================================================

    real(8) function magnetization(spins, L)
        integer, intent(in) :: L
        real(8), intent(in) :: spins(L,L)
        magnetization = sum(spins) / real(L * L, 8)
    end function magnetization
    !====================================================

    function itoa(i) result(str)
        integer, intent(in) :: i
        character(len=12) :: str
        write(str,'(I0)') i
    end function itoa
    !====================================================

end program ising_magnetization_vs_T
