
/*This is a simplified list for the fast version of the program*/


#include <stdio.h>
#include "my_2D_list.h"

void INI_list(List * data, double * vector_dbl, int* vector_int, int dim){
	data->First = (LS_element*)malloc(sizeof(LS_element));
	data->First->LS_value_dbl = vector_dbl;
	data->First->LS_value_int = vector_int;
	data->Last = data->First;
	data->dim = 1;
	for (int i = 1; i < dim; ++i)
	{
		ADD_list(data, vector_dbl + LIST_SIZE_DBL*i, vector_int + LIST_SIZE_INT*i);
	}
};

void DEL_list(List* data, LS_element* previous, LS_element* to_delete){
	
	/*THIS CAUSES PROBLEMS WHEN THERE IS ONLY ONE ELEMENT IN THE LIST!*/
	/*We are deleting the first element*/
	if(previous == NULL){
		data->First = to_delete->LS_next;
	}
	/*We are deleting the last element*/
	if(to_delete->LS_next == NULL){
		data->Last = previous;
		previous->LS_next = NULL;
	}
	/*We are deleting in the middle*/
	if(previous!= NULL && to_delete->LS_next!= NULL){
		previous->LS_next = to_delete->LS_next;
	}

	free(to_delete);
	data->dim -= 1;
};

void ADD_list(List * data, double * value_dbl, int* value_int){
	LS_element * temp;
	temp = data->Last;
	data->Last = (LS_element *) malloc(sizeof(LS_element));
	temp->LS_next = data->Last;
	data->Last->LS_value_dbl = value_dbl;
	data->Last->LS_value_int = value_int;
	data->Last->LS_next  = NULL;
	data->dim += 1;
};

void PRINT_list_dbl(List* data, int row){
	LS_element * temp;
	temp = data->First;
	printf("\n");
	for (int i = 0; i < data->dim; ++i)
	{
		printf("%f, ", temp->LS_value_dbl[row] ) ;
		temp = temp->LS_next;
	}
	printf("\n");
};

void PRINT_list_int(List* data, int row){
	LS_element * temp;
	temp = data->First;
	printf("\n");
	for (int i = 0; i < data->dim; ++i)
	{
		printf("%d     , ", temp->LS_value_int[row] ) ;
		temp = temp->LS_next;
	}
	printf("\n");
};
