! md1d_harmonic_fixed.f90
! Simple 1D molecular dynamics for N atoms connected by harmonic springs
! Compile: gfortran -O2 -o md1d_harmonic md1d_harmonic_fixed.f90
! Run: ./md1d_harmonic

program md1d_harmonic
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer :: i, step, N, nsteps, print_freq
  real(dp) :: dt, kspring, a0, temp, noise_amp
  real(dp), allocatable :: x(:), v(:), f(:), x_old(:)
  real(dp) :: ke, pe, totE
  real(dp) :: rnd

  ! ==== Parameters ====
  N = 1000
  nsteps = 2000
  dt = 0.001_dp
  kspring = 100.0_dp
  a0 = 1.0_dp
  temp = 0.1_dp
  noise_amp = 1.0e-3_dp
  print_freq = 100

  allocate(x(N), v(N), f(N), x_old(N))
  call random_seed()

  ! ==== Initialization ====
  do i = 1, N
    call random_number(rnd)
    x(i) = (i-1)*a0 + noise_amp*(2.0_dp*rnd - 1.0_dp)
  end do

  do i = 1, N
    call random_number(rnd)
    v(i) = sqrt(temp)*(2.0_dp*rnd - 1.0_dp)
  end do

  v = v - sum(v)/real(N, dp)

  call compute_forces(N, x, f, kspring, a0)

  ! ==== Output files ====
  open(unit=10, file='energies.dat', status='replace')
  write(10,'(A)') '# step  kinetic   potential   total'

  open(unit=11, file='positions.dat', status='replace')
  write(11,'(A)') '# step  x1 x2 x3 ... xN'

  call compute_energy(N, x, v, ke, pe, kspring, a0)
  totE = ke + pe
  write(10,'(I8,3F15.6)') 0, ke, pe, totE
  call write_positions(11, 0, x)

  ! ==== Velocity-Verlet main loop ====
  do step = 1, nsteps
    v = v + 0.5_dp * dt * f
    x_old = x
    x = x + dt * v

    call apply_pbc(N, x, a0)
    call compute_forces(N, x, f, kspring, a0)
    v = v + 0.5_dp * dt * f

    if (mod(step, print_freq) == 0) then
      call compute_energy(N, x, v, ke, pe, kspring, a0)
      totE = ke + pe
      write(10,'(I8,3F15.6)') step, ke, pe, totE
      call write_positions(11, step, x)
    end if
  end do

  close(10)
  close(11)
  print *, 'Simulation finished! Files: energies.dat, positions.dat'

contains

  subroutine compute_forces(N, x, f, kspring, a0)
    implicit none
    integer, intent(in) :: N
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)
    real(dp), intent(in) :: kspring, a0
    integer :: i, ip, im
    real(dp) :: dx, L

    L = real(N, dp)*a0
    f = 0.0_dp

    do i = 1, N
      ip = i + 1; if (ip > N) ip = 1
      im = i - 1; if (im < 1) im = N

      dx = x(i) - x(im)
      dx = dx - nint(dx/L)*L
      f(i) = f(i) - kspring*(dx - a0)

      dx = x(ip) - x(i)
      dx = dx - nint(dx/L)*L
      f(i) = f(i) + kspring*(dx - a0)
    end do
  end subroutine compute_forces


  subroutine compute_energy(N, x, v, ke, pe, kspring, a0)
    implicit none
    integer, intent(in) :: N
    real(dp), intent(in) :: x(:), v(:)
    real(dp), intent(out) :: ke, pe
    real(dp), intent(in) :: kspring, a0
    integer :: i, ip
    real(dp) :: dx, L

    L = real(N, dp)*a0
    ke = 0.0_dp; pe = 0.0_dp

    do i = 1, N
      ke = ke + 0.5_dp*v(i)**2
    end do

    do i = 1, N
      ip = i + 1; if (ip > N) ip = 1
      dx = x(ip) - x(i)
      dx = dx - nint(dx/L)*L
      pe = pe + 0.5_dp*kspring*(dx - a0)**2
    end do
  end subroutine compute_energy


  subroutine apply_pbc(N, x, a0)
    implicit none
    integer, intent(in) :: N
    real(dp), intent(inout) :: x(:)
    real(dp), intent(in) :: a0
    integer :: i
    real(dp) :: L

    L = real(N, dp)*a0
    do i = 1, N
      if (x(i) < 0.0_dp) x(i) = x(i) + L
      if (x(i) >= L) x(i) = x(i) - L
    end do
  end subroutine apply_pbc


  subroutine write_positions(unitnum, step, x)
    implicit none
    integer, intent(in) :: unitnum, step
    real(dp), intent(in) :: x(:)
    integer :: i
    write(unitnum,'(I8,1X)', advance='no') step
    do i = 1, size(x)
      write(unitnum,'(F10.5,1X)', advance='no') x(i)
    end do
    write(unitnum,*)
  end subroutine write_positions

end program md1d_harmonic
