C     ABAQUS UMAT subroutine of GRU
C     Function: Reconstruct GRU structure in ABAQUS
C     Developed by Yijing Zhou
C     02/02/2024


C---------------------------------------------------------------------------------------

C     Section 1 : Matrix operation functions

C---------------------------------------------------------------------------------------

C     0.1 Matrix addition subroutine

      SUBROUTINE MATRIX_ADD(A, B, C, M, N)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: A(M, N), B(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: C(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            C(i, j) = A(i, j) + B(i, j)
          ENDDO
        ENDDO

      END SUBROUTINE MATRIX_ADD

C     0.2 Matrix subtraction subroutine

      SUBROUTINE MATRIX_MINUS(A, B, C, M, N)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: A(M, N), B(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: C(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            C(i, j) = A(i, j) - B(i, j)
          ENDDO
        ENDDO

      END SUBROUTINE MATRIX_MINUS

C     0.3 Matrix multiplication subroutine

      SUBROUTINE MATRIX_MULTIPLY(A, B, C, M, N, K)
        INTEGER, INTENT(IN) :: M, N, K
        DOUBLE PRECISION, INTENT(IN) :: A(M, N), B(N, K)
        DOUBLE PRECISION, INTENT(OUT) :: C(M, K)

        INTEGER :: i, j, l

        DO i = 1, M
          DO j = 1, K
          C(i, j) = 0.0
            DO l = 1, N
              C(i, j) = C(i, j) + A(i, l) * B(l, j)
            ENDDO
          ENDDO
        ENDDO

      END SUBROUTINE MATRIX_MULTIPLY

C     0.4 Matrix Hadamard product subroutine

      SUBROUTINE Hadamard_p(A, B, C, M, N)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: A(M, N), B(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: C(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            C(i, j) = A(i, j) * B(i, j)
          ENDDO
        ENDDO

      END SUBROUTINE Hadamard_p

C     0.5 Matrix transposition subroutine

      SUBROUTINE MATRIX_TRANSPOSE(A, B, M, N)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: A(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: B(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            B(j, i) = A(i, j)
          ENDDO
        ENDDO

      END SUBROUTINE MATRIX_TRANSPOSE

C     0.6 tanh function subroutine

      SUBROUTINE MATRIX_TANH(A, B, M, N)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: A(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: B(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            B(i, j) = (EXP(A(i, j)) - EXP(-A(i, j))) / 
     1       (EXP(A(i, j))+ EXP(-A(i, j)))
          ENDDO
        ENDDO

      END SUBROUTINE MATRIX_TANH

C     0.7 Sigmoid function subroutine

      SUBROUTINE MATRIX_SIGM(A, B, M, N)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: A(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: B(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            B(i, j) = 1.0 / (1.0 + EXP(-A(i, j)))
          ENDDO
        ENDDO

      END SUBROUTINE MATRIX_SIGM

C     0.8 Minmaxscaler subroutine

      SUBROUTINE Minmaxscaler(A, B, M, N, max_, min_)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: max_, min_
        DOUBLE PRECISION, INTENT(IN) :: A(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: B(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            B(i, j) = (A(i, j) - min_) / (max_ - min_)
          ENDDO
        ENDDO

      END SUBROUTINE Minmaxscaler

C     0.9 Minmaxscaler inverse subroutine

      SUBROUTINE Minmaxscaler_inv(A, B, M, N, max_, min_)
        INTEGER, INTENT(IN) :: M, N
        DOUBLE PRECISION, INTENT(IN) :: max_, min_
        DOUBLE PRECISION, INTENT(IN) :: A(M, N)
        DOUBLE PRECISION, INTENT(OUT) :: B(M, N)

        INTEGER :: i, j

        DO i = 1, M
          DO j = 1, N
            B(i, j) = (A(i, j) * (max_ - min_)) + min_
          ENDDO
        ENDDO

      END SUBROUTINE Minmaxscaler_inv

C---------------------------------------------------------------------------------------

C     End of Matrix operation function

C---------------------------------------------------------------------------------------





C---------------------------------------------------------------------------------------      

C     Section 2 : SUBROUTINE of trained GRU

C---------------------------------------------------------------------------------------   

      SUBROUTINE TRAINED_GRU(PROPS, STATEV, STRAN, DSTRAN, out_s, 
     & h_t_l0, h_t_l1, h_t_l2, NTENS, D_PROPS, D_STATEV)
        INTEGER, INTENT(IN) :: NTENS, D_PROPS, D_STATEV
        DOUBLE PRECISION, INTENT(IN) :: PROPS(D_PROPS), STATEV(D_STATEV)
     & ,STRAN(NTENS), DSTRAN(NTENS)
        DOUBLE PRECISION, INTENT(OUT) :: out_s(6, 1), h_t_l0(50, 1), 
     & h_t_l1(50, 1), h_t_l2(50, 1)

      INTEGER :: i, j

C     1 Declare

C     1.1 Hyper parameters & min max value

      INTEGER input_size, hidden_size, output_size, num_layers

      DOUBLE PRECISION strain_min, strain_max

C     1.2 Parameters matrix

C      implicit none

      DOUBLE PRECISION w_ir_l0, w_iz_l0, w_in_l0,
     & w_hr_l0, w_hz_l0, w_hn_l0,
     & b_ir_l0, b_iz_l0, b_in_l0,
     & b_hr_l0, b_hz_l0, b_hn_l0,
     & w_ir_l1, w_iz_l1, w_in_l1,
     & w_hr_l1, w_hz_l1, w_hn_l1,
     & b_ir_l1, b_iz_l1, b_in_l1,
     & b_hr_l1, b_hz_l1, b_hn_l1,
     & w_ir_l2, w_iz_l2, w_in_l2,
     & w_hr_l2, w_hz_l2, w_hn_l2,
     & b_ir_l2, b_iz_l2, b_in_l2,
     & b_hr_l2, b_hz_l2, b_hn_l2,
     & w_linear, b_linear, onev

      DIMENSION w_ir_l0(50, 6), w_iz_l0(50, 6), w_in_l0(50, 6),
     & w_hr_l0(50, 50), w_hz_l0(50, 50), w_hn_l0(50, 50),
     & b_ir_l0(50, 1), b_iz_l0(50, 1), b_in_l0(50, 1),
     & b_hr_l0(50, 1), b_hz_l0(50, 1), b_hn_l0(50, 1),
     & w_ir_l1(50, 50), w_iz_l1(50, 50), w_in_l1(50, 50),
     & w_hr_l1(50, 50), w_hz_l1(50, 50), w_hn_l1(50, 50),
     & b_ir_l1(50, 1), b_iz_l1(50, 1), b_in_l1(50, 1),
     & b_hr_l1(50, 1), b_hz_l1(50, 1), b_hn_l1(50, 1),
     & w_ir_l2(50, 50), w_iz_l2(50, 50), w_in_l2(50, 50),
     & w_hr_l2(50, 50), w_hz_l2(50, 50), w_hn_l2(50, 50),
     & b_ir_l2(50, 1), b_iz_l2(50, 1), b_in_l2(50, 1),
     & b_hr_l2(50, 1), b_hz_l2(50, 1), b_hn_l2(50, 1),
     & w_linear(6, 50), b_linear(6, 1), onev(50, 1)

C     1.3 Take transpose

C     In this case, we do not need this step

C     1.4 Define hidden vector / input_v

      DOUBLE PRECISION h_l0, h_l1, h_l2,
     & input_v, input_v_raw

      DIMENSION h_l0(50, 1), h_l1(50, 1), h_l2(50, 1),
     & input_v(6, 1), input_v_raw(6, 1)

C     1.5 GRU layer 0

      DOUBLE PRECISION r_l0,
     & prod_r1_l0, prod_r2_l0,
     & sum_r1_l0, sum_r2_l0, sum_r3_l0,
     & z_l0,
     & prod_z1_l0, prod_z2_l0,
     & sum_z1_l0, sum_z2_l0, sum_z3_l0,
     & n_l0,
     & prod_n1_l0, prod_n2_l0,
     & sum_n1_l0, sum_n2_l0,
     & had_n_l0, sum_n3_l0,
C    & h_t_l0,
     & diff_l0, 
     & had_h1_l0, had_h2_l0,
     & sum_h_l0

      DIMENSION r_l0(50, 1),
     & prod_r1_l0(50, 1), prod_r2_l0(50, 1),
     & sum_r1_l0(50, 1), sum_r2_l0(50, 1), sum_r3_l0(50, 1),
     & z_l0(50, 1),
     & prod_z1_l0(50, 1), prod_z2_l0(50, 1),
     & sum_z1_l0(50, 1), sum_z2_l0(50, 1), sum_z3_l0(50, 1),
     & n_l0(50, 1),
     & prod_n1_l0(50, 1), prod_n2_l0(50, 1),
     & sum_n1_l0(50, 1), sum_n2_l0(50, 1),
     & had_n_l0(50, 1), sum_n3_l0(50, 1),
C    & h_t_l0(50, 1),
     & diff_l0(50, 1), 
     & had_h1_l0(50, 1), had_h2_l0(50, 1),
     & sum_h_l0(50, 1)

C     1.6 GRU layer 1

      DOUBLE PRECISION r_l1,
     & prod_r1_l1, prod_r2_l1,
     & sum_r1_l1, sum_r2_l1, sum_r3_l1,
     & z_l1,
     & prod_z1_l1, prod_z2_l1,
     & sum_z1_l1, sum_z2_l1, sum_z3_l1,
     & n_l1,
     & prod_n1_l1, prod_n2_l1,
     & sum_n1_l1, sum_n2_l1,
     & had_n_l1, sum_n3_l1,
C    & h_t_l1,
     & diff_l1, 
     & had_h1_l1, had_h2_l1,
     & sum_h_l1

      DIMENSION r_l1(50, 1),
     & prod_r1_l1(50, 1), prod_r2_l1(50, 1),
     & sum_r1_l1(50, 1), sum_r2_l1(50, 1), sum_r3_l1(50, 1),
     & z_l1(50, 1),
     & prod_z1_l1(50, 1), prod_z2_l1(50, 1),
     & sum_z1_l1(50, 1), sum_z2_l1(50, 1), sum_z3_l1(50, 1),
     & n_l1(50, 1),
     & prod_n1_l1(50, 1), prod_n2_l1(50, 1),
     & sum_n1_l1(50, 1), sum_n2_l1(50, 1),
     & had_n_l1(50, 1), sum_n3_l1(50, 1),
C    & h_t_l1(50, 1),
     & diff_l1(50, 1), 
     & had_h1_l1(50, 1), had_h2_l1(50, 1),
     & sum_h_l1(50, 1)

C     1.7 GRU layer 2

      DOUBLE PRECISION r_l2,
     & prod_r1_l2, prod_r2_l2,
     & sum_r1_l2, sum_r2_l2, sum_r3_l2,
     & z_l2,
     & prod_z1_l2, prod_z2_l2,
     & sum_z1_l2, sum_z2_l2, sum_z3_l2,
     & n_l2,
     & prod_n1_l2, prod_n2_l2,
     & sum_n1_l2, sum_n2_l2,
     & had_n_l2, sum_n3_l2,
C    & h_t_l2,
     & diff_l2, 
     & had_h1_l2, had_h2_l2,
     & sum_h_l2

      DIMENSION r_l2(50, 1),
     & prod_r1_l2(50, 1), prod_r2_l2(50, 1),
     & sum_r1_l2(50, 1), sum_r2_l2(50, 1), sum_r3_l2(50, 1),
     & z_l2(50, 1),
     & prod_z1_l2(50, 1), prod_z2_l2(50, 1),
     & sum_z1_l2(50, 1), sum_z2_l2(50, 1), sum_z3_l2(50, 1),
     & n_l2(50, 1),
     & prod_n1_l2(50, 1), prod_n2_l2(50, 1),
     & sum_n1_l2(50, 1), sum_n2_l2(50, 1),
     & had_n_l2(50, 1), sum_n3_l2(50, 1),
C    & h_t_l2(50, 1),
     & diff_l2(50, 1), 
     & had_h1_l2(50, 1), had_h2_l2(50, 1),
     & sum_h_l2(50, 1)

C     1.8 Linear layer

      DOUBLE PRECISION prod_lin
C    & ,out_s

      DIMENSION prod_lin(6, 1) 
C    & ,out_s(1)

C     1.9 E & max value

      DOUBLE PRECISION E, m_1, m_2


C     2 Assignment

C     2.1 Hyper parameters

      input_size = PROPS(1)
      hidden_size = PROPS(2)
      output_size = PROPS(3)
      num_layers = PROPS(4)
      E = PROPS(5)

C     2.2 Parameters matrix

C     GRU layer 0

      data w_ir_l0 /
     & /

      data w_iz_l0 /
     & /

      data w_in_l0 /
     & /

      data w_hr_l0 /
     & /

      data w_hz_l0 /
     & /

      data w_hn_l0 /
     & /

      data b_ir_l0 /
     & /

      data b_iz_l0 /
     & /

      data b_in_l0 /
     & /

      data b_hr_l0 /
     & /

      data b_hz_l0 /
     & /

      data b_hn_l0 /
     & /

C     GRU layer 1

      data w_ir_l1 /
     & /

      data w_iz_l1 /
     & /

      data w_in_l1 /
     & /

      data w_hr_l1 /
     & /

      data w_hz_l1 /
     & /

      data w_hn_l1 /
     & /

      data b_ir_l1 /
     & /

      data b_iz_l1 /
     & /

      data b_in_l1 /
     & /

      data b_hr_l1 /
     & /

      data b_hz_l1 /
     & /

      data b_hn_l1 /
     & /

C     GRU layer 2

      data w_ir_l2 /
     & /

      data w_iz_l2 /
     & /

      data w_in_l2 /
     & /

      data w_hr_l2 /
     & /

      data w_hz_l2 /
     & /

      data w_hn_l2 /
     & /

      data b_ir_l2 /
     & /

      data b_iz_l2 /
     & /

      data b_in_l2 /
     & /

      data b_hr_l2 /
     & /

      data b_hz_l2 /
     & /

      data b_hn_l2 /
     & /

C     linear layer

      data w_linear /
     & /

      data b_linear /
     & /

C     onev

      DO i = 1, 50
        onev(i, 1) = 1.0D0
      ENDDO

C     3 Re-construct a trained RNN model

C     3.1 Preprocess

C      Take transpose

C     In this case, we do not need this step

C      Hidden vector / strain vector

      DO i = 1, 50
        h_l0(i, 1) = STATEV(i)
      ENDDO

      DO i = 1, 50
        h_l1(i, 1) = STATEV(i+50)
      ENDDO

      DO i = 1, 50
        h_l2(i, 1) = STATEV(i+100)
      ENDDO

      DO i = 1, 6
        input_v_raw(i, 1) = (STRAN(i) + DSTRAN(i))
      ENDDO

      strain_min = -0.02156124666
      strain_max = 0.02134107083

      CALL Minmaxscaler(input_v_raw, input_v, 6, 1,
     & strain_max, strain_min)

C     3.2 RNN layer 0

      CALL MATRIX_MULTIPLY(w_ir_l0, input_v, prod_r1_l0, 50, 6, 1)
      CALL MATRIX_MULTIPLY(w_hr_l0, h_l0, prod_r2_l0, 50, 50, 1)
      CALL MATRIX_ADD(prod_r1_l0, b_ir_l0, sum_r1_l0, 50, 1)
      CALL MATRIX_ADD(prod_r2_l0, b_hr_l0, sum_r2_l0, 50, 1)
      CALL MATRIX_ADD(sum_r1_l0, sum_r2_l0, sum_r3_l0, 50, 1)
      CALL MATRIX_SIGM(sum_r3_l0, r_l0, 50, 1)

      CALL MATRIX_MULTIPLY(w_iz_l0, input_v, prod_z1_l0, 50, 6, 1)
      CALL MATRIX_MULTIPLY(w_hz_l0, h_l0, prod_z2_l0, 50, 50, 1)
      CALL MATRIX_ADD(prod_z1_l0, b_iz_l0, sum_z1_l0, 50, 1)
      CALL MATRIX_ADD(prod_z2_l0, b_hz_l0, sum_z2_l0, 50, 1)
      CALL MATRIX_ADD(sum_z1_l0, sum_z2_l0, sum_z3_l0, 50, 1)
      CALL MATRIX_SIGM(sum_z3_l0, z_l0, 50, 1)

      CALL MATRIX_MULTIPLY(w_in_l0, input_v, prod_n1_l0, 50, 6, 1)
      CALL MATRIX_MULTIPLY(w_hn_l0, h_l0, prod_n2_l0, 50, 50, 1)
      CALL MATRIX_ADD(prod_n1_l0, b_in_l0, sum_n1_l0, 50, 1)
      CALL MATRIX_ADD(prod_n2_l0, b_hn_l0, sum_n2_l0, 50, 1)
      CALL Hadamard_p(r_l0, sum_n2_l0, had_n_l0, 50, 1)
      CALL MATRIX_ADD(sum_n1_l0, had_n_l0, sum_n3_l0, 50, 1)
      CALL MATRIX_TANH(sum_n3_l0, n_l0, 50, 1)

      CALL MATRIX_MINUS(onev, z_l0, diff_l0, 50, 1)
      CALL Hadamard_p(diff_l0, n_l0, had_h1_l0, 50, 1)
      CALL Hadamard_p(z_l0, h_l0, had_h2_l0, 50, 1)
      CALL MATRIX_ADD(had_h1_l0, had_h2_l0, h_t_l0, 50, 1)

C     3.3 RNN layer 1

      CALL MATRIX_MULTIPLY(w_ir_l1, h_t_l0, prod_r1_l1, 50, 50, 1)
      CALL MATRIX_MULTIPLY(w_hr_l1, h_l1, prod_r2_l1, 50, 50, 1)
      CALL MATRIX_ADD(prod_r1_l1, b_ir_l1, sum_r1_l1, 50, 1)
      CALL MATRIX_ADD(prod_r2_l1, b_hr_l1, sum_r2_l1, 50, 1)
      CALL MATRIX_ADD(sum_r1_l1, sum_r2_l1, sum_r3_l1, 50, 1)
      CALL MATRIX_SIGM(sum_r3_l1, r_l1, 50, 1)

      CALL MATRIX_MULTIPLY(w_iz_l1, h_t_l0, prod_z1_l1, 50, 50, 1)
      CALL MATRIX_MULTIPLY(w_hz_l1, h_l1, prod_z2_l1, 50, 50, 1)
      CALL MATRIX_ADD(prod_z1_l1, b_iz_l1, sum_z1_l1, 50, 1)
      CALL MATRIX_ADD(prod_z2_l1, b_hz_l1, sum_z2_l1, 50, 1)
      CALL MATRIX_ADD(sum_z1_l1, sum_z2_l1, sum_z3_l1, 50, 1)
      CALL MATRIX_SIGM(sum_z3_l1, z_l1, 50, 1)

      CALL MATRIX_MULTIPLY(w_in_l1, h_t_l0, prod_n1_l1, 50, 50, 1)
      CALL MATRIX_MULTIPLY(w_hn_l1, h_l1, prod_n2_l1, 50, 50, 1)
      CALL MATRIX_ADD(prod_n1_l1, b_in_l1, sum_n1_l1, 50, 1)
      CALL MATRIX_ADD(prod_n2_l1, b_hn_l1, sum_n2_l1, 50, 1)
      CALL Hadamard_p(r_l1, sum_n2_l1, had_n_l1, 50, 1)
      CALL MATRIX_ADD(sum_n1_l1, had_n_l1, sum_n3_l1, 50, 1)
      CALL MATRIX_TANH(sum_n3_l1, n_l1, 50, 1)

      CALL MATRIX_MINUS(onev, z_l1, diff_l1, 50, 1)
      CALL Hadamard_p(diff_l1, n_l1, had_h1_l1, 50, 1)
      CALL Hadamard_p(z_l1, h_l1, had_h2_l1, 50, 1)
      CALL MATRIX_ADD(had_h1_l1, had_h2_l1, h_t_l1, 50, 1)
 
C     3.4 RNN layer 2

      CALL MATRIX_MULTIPLY(w_ir_l2, h_t_l1, prod_r1_l2, 50, 50, 1)
      CALL MATRIX_MULTIPLY(w_hr_l2, h_l2, prod_r2_l2, 50, 50, 1)
      CALL MATRIX_ADD(prod_r1_l2, b_ir_l2, sum_r1_l2, 50, 1)
      CALL MATRIX_ADD(prod_r2_l2, b_hr_l2, sum_r2_l2, 50, 1)
      CALL MATRIX_ADD(sum_r1_l2, sum_r2_l2, sum_r3_l2, 50, 1)
      CALL MATRIX_SIGM(sum_r3_l2, r_l2, 50, 1)

      CALL MATRIX_MULTIPLY(w_iz_l2, h_t_l1, prod_z1_l2, 50, 50, 1)
      CALL MATRIX_MULTIPLY(w_hz_l2, h_l2, prod_z2_l2, 50, 50, 1)
      CALL MATRIX_ADD(prod_z1_l2, b_iz_l2, sum_z1_l2, 50, 1)
      CALL MATRIX_ADD(prod_z2_l2, b_hz_l2, sum_z2_l2, 50, 1)
      CALL MATRIX_ADD(sum_z1_l2, sum_z2_l2, sum_z3_l2, 50, 1)
      CALL MATRIX_SIGM(sum_z3_l2, z_l2, 50, 1)

      CALL MATRIX_MULTIPLY(w_in_l2, h_t_l1, prod_n1_l2, 50, 50, 1)
      CALL MATRIX_MULTIPLY(w_hn_l2, h_l2, prod_n2_l2, 50, 50, 1)
      CALL MATRIX_ADD(prod_n1_l2, b_in_l2, sum_n1_l2, 50, 1)
      CALL MATRIX_ADD(prod_n2_l2, b_hn_l2, sum_n2_l2, 50, 1)
      CALL Hadamard_p(r_l2, sum_n2_l2, had_n_l2, 50, 1)
      CALL MATRIX_ADD(sum_n1_l2, had_n_l2, sum_n3_l2, 50, 1)
      CALL MATRIX_TANH(sum_n3_l2, n_l2, 50, 1)

      CALL MATRIX_MINUS(onev, z_l2, diff_l2, 50, 1)
      CALL Hadamard_p(diff_l2, n_l2, had_h1_l2, 50, 1)
      CALL Hadamard_p(z_l2, h_l2, had_h2_l2, 50, 1)
      CALL MATRIX_ADD(had_h1_l2, had_h2_l2, h_t_l2, 50, 1)

C     3.5 Linear layer

      CALL MATRIX_MULTIPLY(w_linear, h_t_l2, prod_lin, 6, 50, 1)
      CALL MATRIX_ADD(prod_lin, b_linear, out_s, 6, 1)

      END SUBROUTINE TRAINED_GRU

C---------------------------------------------------------------------------------------      

C     End of SUBROUTINE of trained GRU

C---------------------------------------------------------------------------------------  





C---------------------------------------------------------------------------------------

C     Section 3 : UMAT of GRU

C---------------------------------------------------------------------------------------

C---------------------------------------------------------------------------------------
C Start of Base code, DO NOT change
C---------------------------------------------------------------------------------------
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     4 JSTEP(4)
C---------------------------------------------------------------------------------------
C End of Base code
C---------------------------------------------------------------------------------------
C---------------------------------------------------------------------------------------
C Start of USER code
C---------------------------------------------------------------------------------------

C     1 Declare and assignment
      
C     1.1 GRU

      DOUBLE PRECISION out_s, h_t_l0, h_t_l1, h_t_l2

      DIMENSION out_s(6, 1), h_t_l0(50, 1), h_t_l1(50, 1), h_t_l2(50, 1)

C     1.2 DDSDDE

      DOUBLE PRECISION ain, ain_hat, ess, ess_hat

      DIMENSION ain(NTENS), ain_hat(NTENS), ess(NTENS), ess_hat(NTENS)

      DOUBLE PRECISION gra

C     1.3 Others

      INTEGER input_size, hidden_size, output_size, num_layers
     & ,D_PROPS, D_STATEV

      DOUBLE PRECISION stress_min, stress_max

      DOUBLE PRECISION E, v, G

      INTEGER i, j, k

      input_size = PROPS(1)
      hidden_size = PROPS(2)
      output_size = PROPS(3)
      num_layers = PROPS(4)
      E = PROPS(5)

      D_PROPS = 5
      D_STATEV = 150

C     2 DDSDDE (approximate differentiation)

      v = 0.2
      G=E/(2.D0*(1.D0+v))

      DO id=1, NDI
        DO jd=1, NDI
          DDSDDE(jd, id)=(E*v)/((1.D0+v)*(1.D0-2.D0*v))
        END DO
        DDSDDE(id, id)=(E*(1.D0-v))/((1.D0+v)*(1.D0-2.D0*v))
      END DO

      DO id=NDI+1, NTENS
        DDSDDE(id, id)=G
      END DO
    
C     3 Prediction

      CALL TRAINED_GRU(PROPS, STATEV, STRAN, DSTRAN, out_s, h_t_l0, 
     & h_t_l1, h_t_l2, NTENS, D_PROPS, D_STATEV) 

C     4 STRESS

      stress_min = -3813.9959202446
      stress_max = 3818.2436687611

      CALL Minmaxscaler_inv(out_s, STRESS, 6, 1, stress_max, stress_min)

C     5 Update STATEV

      DO i = 1, 50
        STATEV(i) = h_t_l0(i, 1)
      ENDDO

      DO i = 1, 50
        STATEV(i+50) = h_t_l1(i, 1)
      ENDDO

      DO i = 1, 50
        STATEV(i+100) = h_t_l2(i, 1)
      ENDDO

C---------------------------------------------------------------------------------------
C End of USER code
C---------------------------------------------------------------------------------------

      RETURN
      END

C---------------------------------------------------------------------------------------      

C     End of UMAT of GRU

C--------------------------------------------------------------------------------------- 
