/*TO COMPILE  WRITE: cl brwre_functions.c my_list.c in msvc*/

#include <stdio.h>

#define DIM_RANDOM 100
#define RESOLUTION 100
#define RESOLU_INV 1e-2
#define DIM_BOX 10
#define RADIUS 2

typedef struct slfv_c_class
{
	/*cur_value is a double of dimension [RESOLUTION, RESOLUTION]*/
	double* cur_value;
	/*choice indicates what movement we are going to do next*/
	int* choice;
	/*count_choice counts at what index of choice we have arrived*/
	int count_choice;

	/*set of neutral events*/
	double* neutral;
	/*count_neutral counts at what index we have arrived*/
	int count_neutral;
	/*set of positive selection events*/
	double* positive;
	/*count_positive counts at what index we have arrived*/
	int count_positive;
	/*set of negative selection events*/
	double* negative;
	/*count_negaive counts at what index we have arrived*/
	int count_negaive;

	/*uniform is a set of uniform random variables on [0,1]*/
	double* uniform;
	int count_uniform;

	/*avrg stores the average around a point*/
	double avrg;
	/*tryal decied whether we perform the change or not*/
	bool tryal;
};

void initialize(slfv_c_class* slfv_c){

}

void do_step(slfv_c_class* slfv_c){
	/*choice == 0 <-> Neutral event*/
	if (slfv_c->choice == 0)
	{
		slfv_c->avrg = average(*(slfv_c->cur_value), *(slfv_c->neutral[count_neutral]))
		if (slfv_c->avrg > slfv_c->uniform[count_uniform])
		{
			for (int i = 0; i < RADIUS; ++i)
			{
				for (int i = 0; i < RADIUS; ++i)
				{
					slfv_c->cur_value[]
				}
			}
		}
		
		/*Go a step forward with the counters*/
		count_choice  += 1;
		count_uniform += 1;
		count_neutral += 1;
	}
	/*choice == 1 <-> Positive selection event*/
	/*choice == 2 <-> Negative selection event*/
};

void average(double* fun, double* location){

	int avrg = 0;
	for (int i = 0; i < RADIUS; ++i)
	{
		for (int j = 0; j < RADIUS; ++j)
		{
			avrg += fun[ location[0]+i, location[1]+j ];
		}
	}
	return avrg/RADIUS**2

};