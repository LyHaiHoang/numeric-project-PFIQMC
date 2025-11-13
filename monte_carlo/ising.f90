PROGRAM Ising2D
    IMPLICIT NONE
    INTEGER, PARAMETER :: L = 100, n_mcs = 1000, n_eq = n_mcs/2
    REAL(8) :: T, H, H0=0.0D0
    REAL(8), PARAMETER :: kb = 1.0D0, Jc = 1.0D0
    INTEGER, DIMENSION(L,L) :: spins
    REAL(8), DIMENSION(:), ALLOCATABLE :: T_vals, mT_vals, mH_vals, H_vals, Lm_vals, Hc_vals
    REAL(8) :: T_start, T_end, dT, m_sum, m_tmp, H_start, H_end, dH, dHc, Lm_tmp, m_up, m_down
    INTEGER :: i, j, kT, nT, n_up, n_down, nH, kH
    CHARACTER(LEN=100) :: filename
    INTEGER :: choix = 3

    ! --- Paramètres T ---
    T_start = 1.0D0
    T_end   = 4.0D0
    dT      = 0.5D0

    nT = CEILING((T_end - T_start)/dT) + 1

    ALLOCATE(T_vals(nT))
    ALLOCATE(mT_vals(nT))
    ALLOCATE(Lm_vals(nT))
    ALLOCATE(Hc_vals(nT))
    DO kT = 1, nT
        T_vals(kT) = T_start + (kT - 1) * dT
    END DO

    PRINT*,"Temperature values:", T_vals

    ! --- Paramètres H ---
    H_start = -3.0D0
    H_end   = 3.0D0
    dH      = 0.1D0

    n_up = CEILING((H_end - H_start)/dH) + 1
    n_down = n_up - 1
    nH = n_up + n_down

    ALLOCATE(H_vals(nH))
    ALLOCATE(mH_vals(nH))

    DO kH = 1, n_up
        H_vals(kH) = H_start + (kH - 1) * dH
    END DO
    DO kH = 1, n_down
        H_vals(n_up + kH) = H_end - kH * dH
    END DO

    PRINT*,"Champ values:", H_vals

    CALL random_seed()
    CALL spin_init(spins, L, -1)
    CALL aimantation(spins, L, m_tmp)
    PRINT*,"Initial magnetization:", m_tmp

    IF (choix ==1) THEN
        DO kT = 1, nT
            T = T_vals(kT)
            m_sum = 0.0D0
            DO i = 1, n_mcs
                CALL monte_carlo_step(spins, L, T, H0, Jc, kb, m_tmp)
                m_sum = m_sum + m_tmp
            END DO
            mT_vals(kT) = m_sum / REAL(n_mcs,8)
            WRITE(*,'(A,F6.3,A,F10.6)') 'T = ', T, '  M = ', mT_vals(kT)
        END DO

        WRITE(filename, '(A,I0,A,I0,A)') 'mT_L', L, '_', n_mcs, '.dat'
        OPEN(UNIT=10, FILE=TRIM(filename), STATUS='REPLACE')
        WRITE(10,*) '# T    M'
        DO kT = 1, nT
            WRITE(10,'(F8.4,2X,F10.6)') T_vals(kT), mT_vals(kT)
        END DO
        CLOSE(10)

    ELSE IF (choix ==2) THEN
        T = 3.2D0
        DO kH = 1, nH
            H = H_vals(kH)
            m_sum = 0.0D0
            DO i =1, n_eq
                CALL monte_carlo_step(spins, L, T, H, Jc, kb, m_tmp)
                m_sum = m_sum + m_tmp
            END DO
            DO i = n_eq, n_mcs
                CALL monte_carlo_step(spins, L, T, H, Jc, kb, m_tmp)
                m_sum = m_sum + m_tmp
            END DO
            mH_vals(kH) = m_sum / REAL((n_mcs),8)
            WRITE(*,'(A,F6.3,A,F10.6)') 'H = ', H, '  M = ', mH_vals(kH)
        END DO

        WRITE(filename, '(A,I0,A,I0,A)') 'mH_L', L, '_', n_mcs, '.dat'
        OPEN(UNIT=10, FILE=TRIM(filename), STATUS='REPLACE')
        WRITE(10,*) '# H    M'
        DO kH = 1, nH
            WRITE(10,'(F8.4,2X,F10.6)') H_vals(kH), mH_vals(kH)
        END DO
        CLOSE(10)
    ELSE IF (choix ==3) THEN
        PRINT*, "Calcul de la chaleur latente Lm en fonction de T"
        DO kT =1, nT
            T = T_vals(kT)
            DO kH =1, nH
                H = H_vals(kH)
                m_sum = 0.0D0
                DO i =1, n_mcs
                    CALL monte_carlo_step(spins, L, T, H, Jc, kb, m_tmp)
                    m_sum = m_sum + m_tmp
                END DO
                mH_vals(kH) = m_sum / REAL(n_mcs,8)
            END DO
            m_up = 0.0D0
            m_down = 0.0D0
            CALL champ_coercitif(H_vals, mH_vals, Hc_vals(kT), m_up, m_down)
            dHc = ABS(Hc_vals(kT)-Hc_vals(kT-1))
            CALL chaleur_latente(T, m_up, m_down, dHc, dT, Lm_tmp)
            Lm_vals(kT) = Lm_tmp
            WRITE(*,'(A,F6.3,A,F10.6)') 'T = ', T, '  Lm = ', Lm_vals(kT)
        END DO
        WRITE(filename, '(A,I0,A,I0,A)') 'LmT_L', L, '_', n_mcs, '.dat'
        OPEN(UNIT=10, FILE=TRIM(filename), STATUS='REPLACE')
        WRITE(10,*) '# T    L(m)'
        DO kT = 1, nT
            WRITE(10,'(F8.4,2X,F10.6)') T_vals(kT), Lm_vals(kT)
        END DO
        CLOSE(10)
    ELSE
        PRINT*, "Choix invalide. Veuillez choisir 1 pour T ou 2 pour H."
    END IF

    DEALLOCATE(T_vals, mT_vals)
    DEALLOCATE(H_vals, mH_vals)
    DEALLOCATE(Lm_vals)
    DEALLOCATE(Hc_vals)

CONTAINS

    SUBROUTINE spin_init(spins, L, mode)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: L, mode
        INTEGER, DIMENSION(L,L), INTENT(OUT) :: spins
        REAL(8) :: r
        INTEGER :: i, j

        IF (mode == 1) THEN
            spins = 1
        ELSE IF (mode == -1) THEN
            spins = -1
        ELSE
            DO i = 1, L
                DO j = 1, L
                    CALL random_number(r)
                    IF (r < 0.5d0) THEN
                        spins(i,j) = 1
                    ELSE
                        spins(i,j) = -1
                    END IF
                END DO
            END DO
        END IF
    END SUBROUTINE spin_init


    SUBROUTINE delta_E(spins, L, ii, jj, Hloc, Jc, dE)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: L, ii, jj
        INTEGER, DIMENSION(L,L), INTENT(IN) :: spins
        REAL(8), INTENT(IN) :: Hloc, Jc
        REAL(8), INTENT(OUT) :: dE
        INTEGER :: ip, im, jp, jm
        REAL(8) :: s, nn

        
        ip = MOD(ii, L) + 1
        im = MOD(ii - 2 + L, L) + 1
        jp = MOD(jj, L) + 1
        jm = MOD(jj - 2 + L, L) + 1

        s  = DBLE(spins(ii, jj))
        nn = DBLE(spins(ip, jj) + spins(im, jj) + spins(ii, jp) + spins(ii, jm))
        dE = 2.0D0 * s * (Jc * nn + Hloc)
    END SUBROUTINE delta_E

    SUBROUTINE monte_carlo_step(spins, L, Tloc, Hloc, Jc, kbloc, mtmp)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: L
        INTEGER, INTENT(INOUT) :: spins(L,L)
        REAL(8), INTENT(OUT) :: mtmp
        REAL(8), INTENT(IN) :: Tloc, Hloc, Jc, kbloc
        INTEGER :: i, j
        REAL(8) :: dE, r
        DO i = 1, L
            DO j = 1, L
                CALL delta_E(spins, L, i, j, Hloc, Jc, dE)
                IF (dE <= 0.0D0) THEN
                    spins(i,j) = -spins(i,j)
                ELSE
                    CALL random_number(r)
                    IF (r < EXP(-dE / (kbloc * Tloc))) THEN
                        spins(i,j) = -spins(i,j)
                    END IF
                END IF
            END DO
        END DO
        mtmp = SUM(spins) / REAL(L * L, 8)
    END SUBROUTINE monte_carlo_step

    SUBROUTINE aimantation(spins, L, m)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: L
        INTEGER, DIMENSION(L,L), INTENT(IN) :: spins
        REAL(8), INTENT(OUT) :: m
        m = SUM(spins) / REAL(L * L, 8)
    END SUBROUTINE aimantation

    SUBROUTINE chaleur_latente(T, m_up, m_down, dH, dT, Lm)
        IMPLICIT NONE
        REAL(8), INTENT(IN) :: T, m_up, m_down, dH, dT
        REAL(8), INTENT(OUT) :: Lm
        Lm = T * (m_up - m_down) * dH / dT
    END SUBROUTINE chaleur_latente

    SUBROUTINE champ_coercitif(H_vals, mH_vals, Hc, m_up, m_down)
        IMPLICIT NONE
        REAL(8), DIMENSION(:), INTENT(IN) :: H_vals, mH_vals
        REAL(8), INTENT(OUT) :: Hc, m_up, m_down
        INTEGER :: i, n, i_cross
        REAL(8) :: eps, m_max, m_min, H1, H2, m1, m2

        n = SIZE(H_vals)
        eps = 0.5 

        DO i=1, n-1
            
            IF ( (H_vals(i)== 0.0D0) .AND. (ABS(mH_vals(i)) < eps) ) THEN
                PRINT*,"Phase paramagnetique"
                m_up = 0.0D0
                m_down = 0.0D0
                Hc = 0.0D0
                RETURN
            ELSE
                PRINT*,"Phase ferromagnetique"
                IF ((mH_vals(i)*mH_vals(i+1) < 0.0D0)) THEN
                    H1 = H_vals(i)
                    H2 = H_vals(i+1)
                    m1 = mH_vals(i)
                    m2 = mH_vals(i+1)
                    Hc = H1 - m1 * (H2 - H1) / (m2 - m1)
                    PRINT*,"Champ coercitif Hc =", Hc
                    i_cross = i
                    m_up = maxval(mH_vals, dim=1, mask=(mH_vals>0))
                    m_down = minval(mH_vals, dim=1, mask=(mH_vals<0))
                    PRINT*,"Magnetisation m_up =", m_up
                    PRINT*,"Magnetisation m_down =", m_down
                    EXIT
                END IF
            END IF
        END DO

    END SUBROUTINE champ_coercitif


END PROGRAM Ising2D
