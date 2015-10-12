#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from matplotlib import pyplot
from matplotlib.patches import Polygon
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


def plot_sq(points, R):
	plt.axhline(y=0, color='k')
	plt.axvline(x=0, color='k')
	plt.axhline(y=R[1], color='r')
	plt.axvline(x=R[0], color='r')
	poly = Polygon(points,edgecolor='none')
 	plt.gca().add_patch(poly)
 	plt.axis([0, R[0]*2, 0, R[1]*2])
 	plt.show()


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
	# points = map(solver, zip([r1, r1], [c1,c2]))+map(solver, zip([r2, r2], [c1,c2]))
	points = [solve(r2, c1), solve(r1, c1), solve(r1,c2), solve(r2,c2)]
	return (points, map(lambda x:check_arrea(R,x), points))


def main():
	angle = 45
	R = (2,2)
	#range_factor = convert offsets to current corner
	range_factor = math.sqrt(R[0]+R[1])
	factor_a =  [range_factor*x/100.0 for x in range(1,500,1)]
	factor_b =  [range_factor*x/100.0 for x in range(1,500,1)]
	plan = [[a,b] for a in factor_a for b in factor_b]
	adapter = lambda x : create_sq(x[0], x[1],angle,R)
	results = zip(plan, map(adapter, plan))
	filtered_results = filter(lambda x: sum(x[1][1])>3, results)
	sorted_results = sorted(filtered_results, key=lambda x:x[0][0]*x[0][1], reverse=True)
	points = sorted_results[0][1][0]
	print("max sqr: ",(sorted_results[0][0][0]*sorted_results[0][0][1])/(range_factor*1.0))

	plot_sq(points, R)
	
	
	# sqrs = map(lambda x:x[0][0]*x[0][1] , filtered_results)
	# sqrs.sort()
	# print(sqrs[-1])
if __name__ == '__main__':
	main()	