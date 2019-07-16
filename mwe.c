/*TO COMPILE  WRITE: cl brwre_functions.c my_list.c in msvc*/

#include <stdio.h>
#include "my_list.h"




int main()
{
	
	double spatial_noise[5] = {0,0,0,0,0};
	double times[24] = {1,2,4,2,6,3,5,7,4,9,1,11,3,6,2,4,5,1,7,8,3,6,4,5};
	
	List data;
	List* data_ptr;
	data_ptr = &data;

	int pos_choice_vector[3][2] =  { {1, 3}, {2,2}, {3,2} };
	double times_vector[3] = {1,0,3};

	/*List * data, double * vector_dbl, int* vector_int, int dim*/
	INI_list(data_ptr, times_vector, &pos_choice_vector[0][0], 3);

	PRINT_list_dbl(data_ptr, 0);
	PRINT_list_int(data_ptr, 0);
	PRINT_list_int(data_ptr, 1);
	printf("________________________________________\n");
	
	for (int i = 0; i < 100; ++i){

		data_ptr->First->LS_next->LS_value_dbl[0] = i;

		PRINT_list_dbl(data_ptr, 0);
		printf("________________________________________\n");

		PRINT_list_dbl(data_ptr, 0);
		printf("________________________________________\n");
		
			
	}

	return 0;
}

