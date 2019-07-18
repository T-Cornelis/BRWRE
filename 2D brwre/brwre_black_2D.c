
/*TO COMPILE  WRITE: cl brwre_functions.c my_list.c in msvc*/

#include <stdio.h>
#include "my_2D_list.h"

/* have to compile together with my_list.c */
/* The next three have to be changed also int he Cython code!*/

#define DIM_RANDOM 100000
#define INI_PART 15000
#define DIM_BOX 150

/*Error we allow for the noise (below we don't consider it)*/
#define MIN_ERROR 0.000000001

typedef struct BRWRE_sample{
	/* Counts how much noise we have used */
	int count;
	double* spatial_noise;
	int total;
	/* Data contains: Time, Location, Choice */
	List* data;
	/* Exponential times of jumps*/
	double* times;
	/* LIST OF POSSIBLE ERROS
	0 - No error
	1 - All particles died.
	2 - Have to generate new random variables.
	*/
	int ERROR_NUMBER;
	/* What time we want to get to */
	double aim;
	/* Have we reached that time? Y = 1, N = 0*/
	int move_on;
	/* Have we reached the end of the list? Y =1, N= 0*/
	int END_reached;
	/*The element we are moving*/
	LS_element* to_move;
	/*The element before the on we are moving*/
	LS_element* previous;
	double cur_particle;
} Brwre;


void INI_brwre(Brwre* particle, double* spat_noise, double* times, List* ini_pos, double aim){
	
	particle->count = 0;
	particle->spatial_noise = spat_noise;
	particle->times = times;
	particle->total = INI_PART;
	particle->data = ini_pos;
	particle->ERROR_NUMBER = 0;
	particle->aim = aim;
	particle->END_reached = 0;
	particle->move_on = 0;
	particle->to_move = ini_pos->First;
	particle->previous = NULL;
	particle->cur_particle = 1;

};


void SET_times(Brwre * particle){

	double* jump_times = particle->times + particle->count;
	double jump_t;
	int choice = 0;
	int loc_x, loc_y;

	for (int i = 1; i < 4; ++i)
	{
		if(jump_times[i] < jump_times[choice]){
			choice = i;
		};
	}

	jump_t = jump_times[choice];
	loc_x = particle->to_move->LS_value_int[0];
	loc_y = particle->to_move->LS_value_int[1];
	double noise_value = particle->spatial_noise[DIM_BOX*loc_x+loc_y];
	
	if (particle->to_move->LS_value_dbl[0] + jump_t < particle->aim){
	
		if( fabs(noise_value) > MIN_ERROR){
			if (jump_times[4]/fabs(noise_value) < jump_t){
				jump_t = jump_times[4]/fabs(noise_value);
				choice = 4;
				if (noise_value < 0){
					choice = 5;
				}
			}
			particle->count += 5;
		}
		else{
			particle->count += 4;
		}

		particle->to_move->LS_value_dbl[0] += jump_t;
		particle->to_move->LS_value_int[2] = choice;
	}
	else{
		particle->move_on = 1;
	};
	
	/*Check if we need to generate new random variables*/
	if (particle->count > DIM_RANDOM-6){
		particle->ERROR_NUMBER = 2;
	}
};


void DO_move(Brwre* particle){
	if (particle->ERROR_NUMBER==0)
	{
		/* We set a new time for the particle */
		
		SET_times(particle);

		/* If the final time has been reached we proceed */ 
		while(particle->move_on == 1 && particle->END_reached == 0 && particle->ERROR_NUMBER == 0){
			particle->move_on = 0;
			if (particle->to_move->LS_next != NULL){
				particle->previous = particle->to_move;
				particle->to_move  = particle->to_move->LS_next;
				particle->cur_particle += 1;
				SET_times(particle);
			}
			else{
				particle->END_reached = 1;
			}	
		}

		/* If we are not at the end of the list, we move the particle */

		if (particle->END_reached == 0){

			int choice;
			choice = particle->to_move->LS_value_int[2];

			/*We move RIGHT*/
			if (choice == 0){
				particle->to_move->LS_value_int[0] = ((particle->to_move->LS_value_int[0]+1)%DIM_BOX + DIM_BOX)%DIM_BOX;
			}
			
			/*We move LEFT*/
			if (choice == 1){
				particle->to_move->LS_value_int[0] = ((particle->to_move->LS_value_int[0]-1)%DIM_BOX + DIM_BOX)%DIM_BOX;
			}

			/*We move UP*/
			if (choice == 2){
				particle->to_move->LS_value_int[1] = ((particle->to_move->LS_value_int[1]+1)%DIM_BOX + DIM_BOX)%DIM_BOX;
			}

			/*We move DOWN*/
			if (choice == 3){
				particle->to_move->LS_value_int[1] = ((particle->to_move->LS_value_int[1]-1)%DIM_BOX + DIM_BOX)%DIM_BOX;
			}

		
			/*We generate a particle*/
			if (choice == 4){
				
				/*Create space for the new particle*/
				double* new_value_dbl;
				int* new_value_int;
				new_value_dbl = (double*)malloc(sizeof(double));
				new_value_int = (int*)malloc(3*sizeof(int));	

				/*Inherit time and space of the old particle*/
				new_value_dbl[0] = particle->to_move->LS_value_dbl[0];
				new_value_int[0] = particle->to_move->LS_value_int[0];
				new_value_int[1] = particle->to_move->LS_value_int[1];
				new_value_int[2] = particle->to_move->LS_value_int[2];

				/*We add the new particle at the end of the list*/
				ADD_list(particle->data, new_value_dbl, new_value_int);
			}

			/*We kill a particle*/
			if (choice == 5){

				LS_element* temp;
				/* We pass on to the next particle */
				if(particle->to_move->LS_next != NULL){
					temp = particle->to_move->LS_next;
				}
				else{
					particle->END_reached = 1;
				}
				/*And kill the old one */

				DEL_list(particle->data, particle->previous, particle->to_move);
				particle->to_move = temp;
				
				/*We set the limit to 1, because we cannot deal with empty lists*/
				if (particle->data->dim == 1){
					particle->ERROR_NUMBER = 1;
				}
			}
		}
	}
};


void REPEAT_move(Brwre* particle){

	while(particle->ERROR_NUMBER == 0 && particle->END_reached == 0){
	
		DO_move(particle);
		
	}
};


