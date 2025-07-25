10 ' SC62015 HARDWARE TEST HARNESS V1.1
20 OPEN "COM1:9600,N,8,1,A,L,&1A,X,N" AS #1
30 CLS
40 PRINT "HARNESS READY. WAITING..."

50 ' --- MAIN COMMAND LOOP ---
60 LINE INPUT #1, C$
70 IF C$="P" THEN GOSUB 1000 : GOTO 40
80 IF C$="L" THEN GOSUB 2000 : GOTO 40
90 IF C$="X" THEN GOSUB 3000 : GOTO 40
95 IF C$="S" THEN GOSUB 5000 : GOTO 40
100 IF C$="R" THEN GOSUB 4000 : GOTO 40
110 PRINT "UNKNOWN CMD: "; C$
120 GOTO 40

1000 ' --- SUB: PING ---
1010 CLS : PRINT "CMD: PING"
1020 PRINT #1, "PONG"
1030 PRINT "PONG"
1040 GOTO 9000

2000 ' --- SUB: LOAD CODE ---
2010 CLS : PRINT "CMD: LOAD"
2020 LINE INPUT #1, ADDR$
2030 LINE INPUT #1, LEN$
2040 A = VAL(ADDR$)
2050 L = VAL(LEN$)
2060 PRINT "ADDR:"; A; " LEN:"; L
2070 PRINT "LOADING: ";
2080 FOR I=1 TO L
2090   LINE INPUT #1, B$
2100   POKE A+I-1, VAL("&H"+B$)
2110   PRINT ".";
2120 NEXT I
2130 PRINT : PRINT "LOAD OK"
2140 GOTO 9000

3000 ' --- SUB: EXECUTE CODE ---
3010 CLS : PRINT "CMD: EXECUTE"
3020 LINE INPUT #1, ADDR$
3030 A = VAL(ADDR$)
3040 PRINT "CALLING "; A; "..."
3050 CALL A
3060 PRINT "EXEC OK"
3070 GOTO 9000

4000 ' --- SUB: READ MEMORY ---
4010 CLS : PRINT "CMD: READ"
4020 LINE INPUT #1, ADDR$
4030 LINE INPUT #1, LEN$
4040 A = VAL(ADDR$)
4050 L = VAL(LEN$)
4060 PRINT "ADDR:"; A; " LEN:"; L
4070 PRINT "DUMP:"
4080 FOR I=1 TO L
4090   B = PEEK(A+I-1)
4100   B$ = HEX$(B)
4110   IF LEN(B$)=1 THEN B$="0"+B$
4120   PRINT #1, B$
4130   PRINT B$; " ";
4140   IF I MOD 8 = 0 THEN PRINT
4150 NEXT I
4160 PRINT : PRINT "READ OK"
4170 GOTO 9000

5000 ' --- SUB: STATUS MESSAGE ---
5010 CLS : PRINT "STATUS:"
5020 LINE INPUT #1, MSG$
5030 PRINT MSG$
5040 GOTO 9000

9000 ' Send final OK signal and return
9010 PRINT #1, "OK"
9020 RETURN

