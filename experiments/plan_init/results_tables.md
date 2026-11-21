
### bands, N = 32  ( coarse 16²: 6.43 s, 11 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 16 | 7 | 0 | 4.97 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 1.42 |
| zero | 0.002 | 0 | 390 | 1 | 0 | 0.002 | 0.03 | yes | 24 | 20 | 0.00027 | 2.99 |
| cubic | 0 | 57 | 154 | 5 | 14 | 0.00012 | 7.11 | yes | 33 | 28 | 0.00038 | 3.96 |
| cubic (sdot may swap) | – | – | – | – | – | – | 7.11 | yes | 24 | 20 | 0.00027 | 3.01 |
| local plan | 0 | 3 | 29 | 4.2 | 4 | 0.0014 | 6.94 | yes | 10 | 6 | 0.011 | 0.93 |
| local plan (sdot may swap) | – | – | – | – | – | – | 6.94 | yes | 10 | 6 | 0.011 | 0.93 |
| global Knothe | 0 | 255 | 363 | 18 | 17 | 0.0001 | 0.97 | yes | 39 | 34 | 1.5e-05 | 6.42 |

### bands, N = 64  ( coarse 32²: 4.84 s, 16 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 21 | 10 | 0 | 4.82 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 1.85 |
| zero | 0.002 | 0 | 1804 | 1 | 0 | 0.002 | 0.03 | yes | 87 | 82 | 3.1e-05 | 10.45 |
| cubic | 0 | 340 | 644 | 9.1 | 18 | 0.0001 | 5.37 | yes | 37 | 33 | 3.4e-06 | 4.04 |
| cubic (sdot may swap) | – | – | – | – | – | – | 5.37 | yes | 87 | 82 | 3.1e-05 | 10.63 |
| local plan | 0 | 119 | 266 | 12 | 12 | 0.00012 | 5.65 | yes | 25 | 21 | 0.0053 | 2.05 |
| local plan (sdot may swap) | – | – | – | – | – | – | 5.65 | yes | 87 | 82 | 3.1e-05 | 10.70 |
| global Knothe | 0 | 1427 | 1842 | 55 | 200 | 0 | 14.24 | NO | 1 | 0 | 1 | 1.06 |

### sine, N = 32  ( coarse 16²: 6.66 s, 11 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 12 | 6 | 0 | 4.22 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | yes | 11 | – | – | 0.77 |
| zero | 0.0061 | 0 | 272 | 4.5 | 0 | 0.0061 | 0.03 | yes | 11 | 7 | 0.072 | 1.01 |
| cubic | 0 | 140 | 188 | 5.8 | 23 | 0.00096 | 7.56 | yes | 7 | 3 | 0.25 | 0.63 |
| cubic (sdot may swap) | – | – | – | – | – | – | 7.56 | yes | 11 | 7 | 0.072 | 1.04 |
| local plan | 0 | 12 | 14 | 1.9 | 4 | 0.0054 | 7.22 | yes | 5 | 2 | 0.25 | 0.44 |
| local plan (sdot may swap) | – | – | – | – | – | – | 7.22 | yes | 5 | 2 | 0.25 | 0.41 |
| global Knothe | 0 | 146 | 428 | 46 | 200 | 0 | 12.70 | NO | 1 | 0 | 1 | 0.84 |

### sine, N = 64  ( coarse 32²: 3.99 s, 12 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 12 | 7 | 0 | 3.64 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | yes | 16 | – | – | 1.86 |
| zero | 0.0061 | 0 | 1136 | 4.9 | 0 | 0.0061 | 0.02 | yes | 15 | 11 | 0.037 | 0.91 |
| cubic | 0 | 216 | 306 | 3.9 | 14 | 0.0011 | 4.42 | yes | 7 | 3 | 0.25 | 0.37 |
| cubic (sdot may swap) | – | – | – | – | – | – | 4.42 | yes | 7 | 3 | 0.25 | 0.41 |
| local plan | 0 | 63 | 79 | 2.6 | 8 | 0.002 | 4.84 | yes | 5 | 2 | 0.25 | 0.26 |
| local plan (sdot may swap) | – | – | – | – | – | – | 4.84 | yes | 5 | 2 | 0.25 | 0.26 |
| global Knothe | 0 | 866 | 2016 | 1.4e+02 | 200 | 0 | 13.30 | NO | 1 | 0 | 1 | 1.03 |

### star, N = 32  ( coarse 16²: 15.46 s, 25 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 32 | 16 | 0 | 9.94 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 1.39 |
| zero | 0.0047 | 0 | 752 | 3.7 | 0 | 0.0047 | 0.03 | yes | 143 | 139 | 5.5e-05 | 17.89 |
| cubic | 0 | 12 | 26 | 5.4 | 6 | 0.0016 | 15.72 | yes | 9 | 6 | 0.1 | 0.84 |
| cubic (sdot may swap) | – | – | – | – | – | – | 15.72 | yes | 9 | 6 | 0.1 | 0.81 |
| local plan | 0.01 | 0 | 6 | 1.8 | 0 | 0.01 | 15.91 | yes | 7 | 4 | 0.015 | 0.60 |
| local plan (sdot may swap) | – | – | – | – | – | – | 15.91 | yes | 7 | 4 | 0.015 | 0.60 |
| global Knothe | 0 | 146 | 224 | 37 | 200 | 0 | 15.38 | NO | 1 | 0 | 1 | 1.03 |

### star, N = 64  ( coarse 32²: 10.24 s, 32 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 43 | 22 | 0 | 11.06 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 1.93 |
| zero | 0.0047 | 0 | 3136 | 3.7 | 0 | 0.0047 | 0.03 | NO | 200 | 200 | 4.8e-07 | 50.71 |
| cubic | 0 | 82 | 108 | 11 | 200 | 0 | 19.55 | NO | 3 | 2 | 0.25 | 5.18 |
| cubic (sdot may swap) | – | – | – | – | – | – | 19.55 | NO | 200 | 200 | 4.8e-07 | 50.64 |
| local plan | 0 | 31 | 42 | 15 | 14 | 0.00024 | 11.73 | yes | 15 | 11 | 0.021 | 1.49 |
| local plan (sdot may swap) | – | – | – | – | – | – | 11.73 | NO | 200 | 200 | 4.8e-07 | 50.89 |
| global Knothe | 0 | 774 | 1014 | 1.8e+02 | 200 | 0 | 23.12 | NO | 1 | 0 | 1 | 1.03 |

### bands, N = 128  ( coarse 64²: 5.30 s, 21 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 28 | 14 | 0 | 6.72 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 5.34 |
| zero | 0.002 | 0 | 7708 | 1 | 0 | 0.002 | 0.03 | NO | 200 | 200 | 3.8e-06 | 104.38 |
| cubic | 0 | 1709 | 2372 | 15 | 200 | 0 | 28.58 | NO | 1 | 0 | 1 | 1.11 |
| cubic (sdot may swap) | – | – | – | – | – | – | 28.58 | NO | 200 | 200 | 3.8e-06 | 97.91 |
| local plan | 0 | 704 | 1138 | 24 | 26 | 0.0001 | 7.23 | yes | 72 | 68 | 8.6e-06 | 7.75 |
| local plan (sdot may swap) | – | – | – | – | – | – | 7.23 | NO | 200 | 200 | 3.8e-06 | 101.31 |
| global Knothe | 0 | 5925 | 7778 | 2.1e+02 | 200 | 0 | 22.36 | NO | 1 | 0 | 1 | 1.05 |

### bands, N = 256  ( coarse 128²: 6.08 s, 28 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 40 | 21 | 0 | 9.07 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 9.06 |
| zero | 0.002 | 0 | 31720 | 1 | 0 | 0.002 | 0.04 | NO | 200 | 200 | 2.4e-07 | 225.98 |
| cubic | 0 | 5717 | 6664 | 26 | 200 | 0 | 30.84 | NO | 1 | 0 | 1 | 7.32 |
| cubic (sdot may swap) | – | – | – | – | – | – | 30.84 | NO | 200 | 200 | 2.4e-07 | 223.67 |
| local plan | 0 | 4021 | 4859 | 45 | 200 | 0 | 28.62 | NO | 1 | 0 | 1 | 7.87 |
| local plan (sdot may swap) | – | – | – | – | – | – | 28.62 | NO | 200 | 200 | 2.4e-07 | 220.92 |
| global Knothe | 0 | 23069 | 31830 | 5.7e+02 | 200 | 0 | 23.82 | NO | 1 | 0 | 1 | 1.33 |

### sine, N = 128  ( coarse 64²: 3.65 s, 12 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 16 | 9 | 0 | 4.16 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | yes | 40 | – | – | 7.95 |
| zero | 0.0061 | 0 | 4752 | 5 | 0 | 0.0061 | 0.03 | yes | 21 | 18 | 0.023 | 1.52 |
| cubic | 0 | 406 | 532 | 5.1 | 18 | 0.00012 | 4.37 | yes | 8 | 4 | 0.041 | 0.62 |
| cubic (sdot may swap) | – | – | – | – | – | – | 4.37 | yes | 21 | 18 | 0.023 | 1.61 |
| local plan | 0 | 64 | 127 | 1.4 | 6 | 0.0009 | 5.20 | yes | 5 | 2 | 0.25 | 0.36 |
| local plan (sdot may swap) | – | – | – | – | – | – | 5.20 | yes | 5 | 2 | 0.25 | 0.63 |
| global Knothe | 0 | 4127 | 8470 | 4e+02 | 200 | 0 | 22.43 | NO | 1 | 0 | 1 | 1.05 |

### sine, N = 256  ( coarse 128²: 4.09 s, 16 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 18 | 13 | 0 | 5.23 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | yes | 77 | – | – | 20.32 |
| zero | 0.0061 | 0 | 18960 | 5 | 0 | 0.0061 | 0.04 | yes | 42 | 38 | 0.0019 | 4.97 |
| cubic | 0 | 686 | 814 | 6.4 | 18 | 0.00026 | 5.25 | yes | 10 | 6 | 0.11 | 1.14 |
| cubic (sdot may swap) | – | – | – | – | – | – | 5.25 | yes | 42 | 38 | 0.0019 | 6.41 |
| local plan | 0 | 75 | 129 | 5.2 | 15 | 0.00034 | 9.82 | yes | 6 | 3 | 0.25 | 0.73 |
| local plan (sdot may swap) | – | – | – | – | – | – | 9.82 | yes | 42 | 38 | 0.0019 | 6.97 |
| global Knothe | 0 | 18344 | 34949 | 9e+02 | 200 | 0 | 27.92 | NO | 1 | 0 | 1 | 1.51 |

### star, N = 128  ( coarse 64²: 11.18 s, 43 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 54 | 34 | 0 | 14.47 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 4.64 |
| zero | 0.0047 | 0 | 12750 | 3.7 | 0 | 0.0047 | 0.03 | NO | 200 | 200 | 1.9e-06 | 60.05 |
| cubic | 0 | 152 | 244 | 16 | 17 | 0.00012 | 12.10 | yes | 20 | 17 | 0.0095 | 3.04 |
| cubic (sdot may swap) | – | – | – | – | – | – | 12.10 | NO | 200 | 200 | 1.9e-06 | 61.61 |
| local plan | 0 | 189 | 266 | 33 | 200 | 0 | 36.34 | NO | 1 | 0 | 1 | 1.24 |
| local plan (sdot may swap) | – | – | – | – | – | – | 36.34 | NO | 200 | 200 | 1.9e-06 | 61.11 |
| global Knothe | 0 | 3331 | 4347 | 7.1e+02 | 200 | 0 | 24.82 | NO | 1 | 0 | 1 | 1.23 |

### star, N = 256  ( coarse 128²: 14.60 s, 54 it )

| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdot default (cloud path) | – | – | – | – | – | – | 0.00 | yes | 69 | 48 | 0 | 17.51 |
| sdot default (grid path) | – | – | – | – | – | – | 0.00 | NO | 0 | – | – | 9.16 |
| zero | 0.0047 | 0 | 51420 | 3.7 | 0 | 0.0047 | 0.04 | NO | 200 | 200 | 9.5e-07 | 144.54 |
| cubic | 0 | 752 | 958 | 61 | 200 | 0 | 39.56 | NO | 2 | 1 | 0.25 | 16.95 |
| cubic (sdot may swap) | – | – | – | – | – | – | 39.56 | NO | 200 | 200 | 9.5e-07 | 147.18 |
| local plan | 0 | 1870 | 2126 | 81 | 200 | 0 | 61.95 | NO | 2 | 1 | 0.5 | 7.69 |
| local plan (sdot may swap) | – | – | – | – | – | – | 61.95 | NO | 200 | 200 | 9.5e-07 | 149.90 |
| global Knothe | 0 | 13830 | 18057 | 1.4e+03 | 200 | 0 | 29.37 | NO | 1 | 0 | 1 | 1.41 |
