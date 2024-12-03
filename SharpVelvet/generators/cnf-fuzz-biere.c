/*****************************************************************************
Copyright (c) 2009, Armin Biere, Johannes Kepler University.
Copyright (c) 2024, Anna L.D. Latour and Mate Soos.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <cstdlib>

#define MAX 50
static int clause[MAX + 1];

static int
pick (int from, int to)
{
  assert (from <= to);
  return (rand() % (to - from + 1)) + from;
}

int
main (int argc, char ** argv)
{
  int seed, nlayers, ** layers, *width, * low, * high, * clauses;
  int i, j, k, l, m, n, o, p, sign, lit, layer, w;
  int ** unused, * nunused;
  char * mark;

  if (argc > 1 && !strcmp (argv[1], "-h"))
    {
      printf (
        "usage: cnfuzz [-h][<seed>]\n"
	"\n"
	"If the seed is not specified it is calculated from the process id\n"
	"and the current system time (in seconds).\n");
      return 0;
    }

  seed = (argc > 1) ? atoi (argv[1]) : std::abs ((time(NULL)) >> 1);
  int type = (argc > 2) ? atoi (argv[2]) : 0;
  /* printf ("c seed %d\n", seed); */
  srand (seed);
  w = pick (20, 50);
  /* printf ("c max width %d\n", w); */
  nlayers = pick (1, 2);
  /* printf ("c layers %d\n", nlayers); */
  layers = (int**)calloc (nlayers, sizeof *layers);
  width = (int*)calloc (nlayers, sizeof *width);
  low = (int*)calloc (nlayers, sizeof *low);
  high = (int*)calloc (nlayers, sizeof *high);
  clauses = (int*)calloc (nlayers, sizeof *clauses);
  unused = (int**)calloc (nlayers, sizeof *unused);
  nunused = (int*)calloc (nlayers, sizeof *nunused);
  for (i = 0; i < nlayers; i++)
    {
      width[i] = pick (10, w);
      low[i] = i ? high[i-1] + 1 : 1;
      high[i] = low[i] + width[i] - 1;
      m = width[i];
      if (i) m += width[i-1];
      n = (pick (300, 450) * m) / 100;
      clauses[i] = n;
      /* printf ("c layer[%d] = [%d..%d] w=%d v=%d c=%d r=%.2f\n", */
      /*         i, low[i], high[i], width[i], m, n, n / (double) m); */

      nunused[i] = 2 * (high[i] - low[i] + 1);
      unused[i] = (int*)calloc (nunused[i], sizeof *unused[i]);
      k = 0;
      for (j = low[i]; j <= high[i]; j++)
	for (sign = -1; sign <= 1; sign += 2)
	  unused[i][k++] = sign * j;
      assert (k == nunused[i]);
    }
  n = 0;
  m = high[nlayers-1];
  mark = (char*)calloc (m + 1, 1);
  for (i = 0; i < nlayers; i++)
    n += clauses[i];
  printf ("p cnf %d %d\n", m, n);
  if (type == 0) printf ("c t mc\n");
  else if (type == 1) printf ("c t pmc\n");
  else if (type == 2) printf ("c t wmc\n");
  else if (type == 3) printf ("c t pwmc\n");
  for (i = 0; i < nlayers; i++)
    {
      for (j = 0; j < clauses[i]; j++)
	{
	  l = 3;
	  while (l < MAX && pick (17, 19) != 17)
	    l++;

	  for (k = 0; k < l; k++)
	    {
	      layer = i;
	      while (layer && pick (3, 4) == 3)
		layer--;
	      if (nunused[layer])
		{
		  o = nunused[layer] - 1;
		  p = pick (0, o);
		  lit = unused[layer][p];
		  if (mark [abs (lit)]) continue;
		  nunused[layer] = o;
		  if (p != o) unused[layer][p] = unused[layer][o];
		}
	      else
		{
		  lit = pick (low[layer], high[layer]);
		  if (mark[lit]) continue;
		  sign = (pick (31, 32) == 31) ? 1 : -1;
		  lit *= sign;
		}
	      clause[k] = lit;
	      mark[abs (lit)] = 1;
	      printf ("%d ", lit);
	    }
	  printf ("0\n");
	  for (k = 0; k < l; k++)
	    mark[abs (clause[k])] = 0;
	}
    }
  free (mark);
  free (clauses);
  free (high);
  free (low);
  free (width);
  free (nunused);
  for (i = 0; i < nlayers; i++)
    free (layers[i]), free (unused[i]);
  free (layers);
  return 0;
}
