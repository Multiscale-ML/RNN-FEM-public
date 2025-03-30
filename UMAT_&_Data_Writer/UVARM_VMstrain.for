C     ABAQUS UVARM subroutine of Von Mises strain
C     Function: Von Mises strain visualization
C     Developed by Yijing Zhou
C     02/02/2024


C---------------------------------------------------------------------------------------

C     Section: UVARM

C---------------------------------------------------------------------------------------

      SUBROUTINE UVARM(UVAR,DIRECT,T,TIME,DTIME,CMNAME,ORNAME,
     & NUVARM,NOEL,NPT,LAYER,KSPT,KSTEP,KINC,NDI,NSHR,COORD,
     & JMAC,JMATYP,MATLAYO,LACCFLA)
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3 FLGRAY(15)
      DIMENSION UVAR(NUVARM),DIRECT(3,3),T(3,3),TIME(2)
      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),COORD(*)

C     Error counter:
      JERROR = 0

C     Retrieve the total strain components at the material point
C     The variable ID for total strains in ABAQUS is usually "E"
      CALL GETVRM('E', ARRAY, JARRAY, FLGRAY, JRCD, JMAC, JMATYP,
     & MATLAYO, LACCFLA)

      JERROR = JERROR + JRCD

C     Calculate Von Mises strain based on the retrieved strain components
C     STRAIN(1) = epsilon_xx, STRAIN(2) = epsilon_yy, STRAIN(3) = epsilon_zz
C     STRAIN(4) = epsilon_xy, STRAIN(5) = epsilon_yz, STRAIN(6) = epsilon_zx
      UVAR(1) = SQRT((1.0/2.0)*((ARRAY(1)-ARRAY(2))**2 + 
     & (ARRAY(2)-ARRAY(3))**2 + (ARRAY(3)-ARRAY(1))**2 +
     & 6.0*(ARRAY(4)**2 + ARRAY(5)**2 + ARRAY(6)**2)))

C     If error, write comment to .DAT file:
      IF(JERROR.NE.0)THEN
        WRITE(6,*) 'REQUEST ERROR IN UVARM FOR ELEMENT NUMBER ',
     &      NOEL,'INTEGRATION POINT NUMBER ',NPT
      ENDIF

      RETURN
      END

C---------------------------------------------------------------------------------------      

C     End of UVARM

C---------------------------------------------------------------------------------------
