print('HELLO JIAMIN')

import matplotlib
matplotlib.use('Agg')
import pylab
pylab.plot([1,2], [3,4], linestyle='-')
pylab.savefig('foo.png')
