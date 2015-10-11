#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import math

#(a1, a2, a3)
def get_coeffs(angle, R):
	if(angle == 90):
		return (0, 1, -R[1])
	offset =  math.tan(math.radians(angle))*R[1]
	# print("offset ",offset)
	if (offset == 0):
		return (1, 0, -R[0])
	x1 = R[0]; y1 = R[1]; x2 = R[0]+offset; y2=0;
	return (y1-y2, x2-x1, x1*y2-x2*y1)

def convert_to_f(coeffs, offset=0):
	return lambda x: (1.0 / coeffs[1])*(-coeffs[0]*x -coeffs[2])


def check_arrea(R, point):
	if ( ( (point[0]<=R[0])&(point[0]>=0)&(point[1]>=0)) or ((point[0]>=R[0])&(point[1]>=0)&(point[1]<=R[1]) ) ):
		return True
	return False


def create_sq(h, w, angle, R, flag=False):
	def solve(eq1, eq2):
		a = np.array([[eq1[0], eq1[1]], [eq2[0], eq2[1]]])
		b = np.array([-eq1[2], -eq2[2]])
		x = np.linalg.solve(a, b)
		return [x[0], x[1]]

	r1 = get_coeffs(angle, (R[0],R[1]))
	r2 = get_coeffs(angle, (R[0]-h/2.0,R[1]-h/2.0))
	
	c1 = get_coeffs(90 + angle, (R[0]+w/2,R[1]))
	c2 = get_coeffs(90 + angle, (R[0]-w/2,R[1]))

	solver = lambda x:solve(x[0], x[1])
	points = map(solver, zip([r1, r1], [c1,c2]))+map(solver, zip([r2, r2], [c1,c2]))
	
	if(flag):
		# plt.grid(True, which='both')
		plt.axhline(y=0, color='k')
		plt.axvline(x=0, color='k')
		plt.axhline(y=R[1], color='r')
		plt.axvline(x=R[0], color='r')
	
		plt.plot(map(convert_to_f(r1), range(-5,5)),range(-5,5),color="b")
		plt.plot(map(convert_to_f(r2), range(-5,5)),range(-5,5),color="b")
		plt.plot(map(convert_to_f(c1), range(-5,5)),range(-5,5),color="b")
		plt.plot(map(convert_to_f(c2), range(-5,5)),range(-5,5),color="b")
		plt.axis([-0.5, 2, -0.5, 2])

		plt.show()

	return map(lambda x:check_arrea(R,x), points)


def main():
	angle = 45
	R = (1,1)
	factor_a =  [x/100.0 for x in range(1,500,1)]
	factor_b =  [x/100.0 for x in range(1,500,1)]
	plan = [[a,b] for a in factor_a for b in factor_b]
	adapter = lambda x : create_sq(x[0], x[1],angle,R)
	results = zip(plan, map(adapter, plan))
	filtered_results = filter(lambda x: sum(x[1])>3, results)
	for x in map(lambda x:sum(x[0]) , filtered_results):
		print(x)
if __name__ == '__main__':
	main()	