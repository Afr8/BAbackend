import numpy as np

def zhang_and_suen_binary_thinning(A):
	height = A.shape[0]
	width = A.shape[1]

	_A = np.copy(A)
	
	removed_points = []
	flag_removed_point = True

	while flag_removed_point:

		flag_removed_point = False

		for x in range(1, height - 1):
			for y in range(1, width - 1):
				if _A[x,y,0] == 1:
					#get 8-neighbors
					neighborhood = [_A[x-1, y,0], _A[x-1, y+1,0], _A[x, y+1,0], _A[x+1, y+1,0], _A[x+1, y,0], _A[x+1, y-1,0], _A[x, y-1,0], _A[x-1, y-1,0]]
					P2, P3, P4, P5, P6, P7, P8, P9 = neighborhood

					#B_P1 is the number of nonzero neighbors of P1=(x, y)
					B_P1 = np.sum(neighborhood)
					condition_1 = 2 <= B_P1 <= 6
				
					#A_P1 is the number of 01 patterns in the ordered set of neighbors
					n = neighborhood + neighborhood[0:1]
					A_P1 = sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

					condition_2 = A_P1 == 1
					condition_3 = P2 * P4 * P6 == 0
					condition_4 = P4 * P6 * P8 == 0

					if(condition_1 and condition_2 and condition_3 and condition_4):
						removed_points.append((x, y))
						flag_removed_point = True

		for x, y in removed_points:
			_A[x, y] = 0
		del removed_points[:]

		for x in range(1, height - 1):
			for y in range(1, width - 1):

				if _A[x,y,0] == 1:
					#get 8-neighbors
					neighborhood =  [_A[x-1, y,0], _A[x-1, y+1,0], _A[x, y+1,0], _A[x+1, y+1,0], _A[x+1, y,0], _A[x+1, y-1,0], _A[x, y-1,0], _A[x-1, y-1,0]]
					P2, P3, P4, P5, P6, P7, P8, P9 = neighborhood

					#B_P1 is the number of nonzero neighbors of P1=(x, y)
					B_P1 = np.sum(neighborhood)
					condition_1 = 2 <= B_P1 <= 6
				
					#A_P1 is the number of 01 patterns in the ordered set of neighbors
					n = neighborhood + neighborhood[0:1]
					A_P1 = sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

					condition_2 = A_P1 == 1			
					condition_3 = P2 * P4 * P8 == 0
					condition_4 = P2 * P6 * P8 == 0
				
					if(condition_1 and condition_2 and condition_3 and condition_4):
						removed_points.append((x, y))
						flag_removed_point = True

		for x, y in removed_points:
			_A[x, y] = 0
		del removed_points[:]

	output = _A

	return output

