program ising_magnetization_vs_T
  implicit none
  integer, parameter :: L = 100, n_mcs = 100
  integer, parameter :: N = L*L
  real(8), parameter :: kb = 1.0d0, J = 1.0d0, H = 0.0d0
  real(8), dimension(L,L) :: spins
  real(8), dimension(:), allocatable :: T_vals, mT_vals
  real(8) :: T, T_start, T_end, dT
  integer :: iT, i, j, step
  character(len=100) :: filename

  call random_seed()

  ! === Paramètres du balayage en température ===
  T_start = 1.0d0
  T_end   = 4.0d0
  dT      = 0.2d0
  allocate(T_vals(int((T_end - T_start)/dT) + 1))
  allocate(mT_vals(size(T_vals)))

  ! === Remplissage du vecteur des températures ===
  do iT = 1, size(T_vals)
     T_vals(iT) = T_start + dT*(iT - 1)
  end do

  ! === Initialisation des spins ===
  call spin_init(spins, L, 1)

  print *, "=== Simulation M(T) du modèle d’Ising 2D ==="

  ! === Boucle sur les températures ===
  do iT = 1, size(T_vals)
     T = T_vals(iT)
     print *, "Température: ", T

     ! Thermalisation (atteindre l’équilibre)
     do step = 1, n_mcs
        call metropolis(spins, L, H, J, kb, T)
     end do

     ! Calcul de la magnétisation moyenne
     mT_vals(iT) = aimantation(spins, L)
  end do

  ! === Sauvegarde des résultats ===
  filename = "mT_L" // trim(adjustl(itoa(L))) // "_step" // trim(adjustl(itoa(n_mcs))) // "_H0.dat"
  open(unit=10, file=filename, status="replace")
  write(10,'(A)') "# T(K)   M(T)"
  do iT = 1, size(T_vals)
     write(10,'(2F12.6)') T_vals(iT), mT_vals(iT)
  end do
  close(10)
  print *, "Résultats sauvegardés dans ", trim(filename)

contains

  subroutine spin_init(spins, L, mode)
    integer, intent(in) :: L, mode
    real(8), intent(out) :: spins(L,L)
    if (mode == 1) then
       spins = 1.0d0
    else
       spins = -1.0d0
    end if
  end subroutine spin_init

  real(8) function delta_E(spins, L, i, j, H, J)
    integer, intent(in) :: L, i, j
    real(8), intent(in) :: spins(L,L), H, J
    integer :: ip, im, jp, jm
    real(8) :: s, nn
    ip = mod(i, L) + 1
    im = mod(i - 2, L) + 1
    jp = mod(j, L) + 1
    jm = mod(j - 2, L) + 1
    s = spins(i, j)
    nn = spins(ip, j) + spins(im, j) + spins(i, jp) + spins(i, jm)
    delta_E = 2.0d0 * s * (J * nn + H)
  end function delta_E

  subroutine metropolis(spins, L, H, J, kb, T)
    integer, intent(in) :: L
    real(8), intent(inout) :: spins(L,L)
    real(8), intent(in) :: H, J, kb, T
    real(8) :: dE, beta, r
    integer :: k, i, j
    beta = 1.0d0 / (kb * T)

    do k = 1, L*L
       call random_number(r)
       i = 1 + int(r * L)
       call random_number(r)
       j = 1 + int(r * L)
       dE = delta_E(spins, L, i, j, H, J)
       call random_number(r)
       if (dE <= 0.0d0 .or. r < exp(-beta * dE)) then
          spins(i, j) = -spins(i, j)
       end if
    end do
  end subroutine metropolis

  real(8) function aimantation(spins, L)
    integer, intent(in) :: L
    real(8), intent(in) :: spins(L,L)
    aimantation = sum(spins) / real(L*L, 8)
  end function aimantation

  function itoa(i) result(str)
    integer, intent(in) :: i
    character(len=12) :: str
    write(str,'(I0)') i
  end function itoa

end program ising_magnetization_vs_T
