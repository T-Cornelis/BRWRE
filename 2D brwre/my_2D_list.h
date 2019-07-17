#ifndef MY_2D_LIST_H
#define MY_2D_LIST_H

#define LIST_SIZE_DBL 1
#define LIST_SIZE_INT 3

typedef struct List_Element{
	/*Time*/
	double* LS_value_dbl;
	/*Location, Choice*/
	int* LS_value_int;
	struct List_Element * LS_next;
} LS_element;

typedef struct List{
	LS_element * First;
	LS_element * Last;
	size_t dim;
}List;

/*extern void ADD_element(LS_element* previous, double *value_dbl, int* value_int);
extern LS_element* GET_element(List* data, size_t lim);*/
extern void DEL_list(List* data, LS_element* previous, LS_element* to_delete);
extern void ADD_list(List * data, double* value_dbl, int* value_int);
extern void INI_list(List * data, double * vector_dbl, int* vector_int, int dim);
/*extern void PRINT_list(List* data, int row);
extern LS_element* LS_arg_min(List* data, int row, LS_element* previous);
extern void ADD_list_ordered(List * data, double * value_dbl, int* value_int);
extern void DEL_list_ordered(List* data);
extern void REORDER_list(List* data);*/
extern void PRINT_list_dbl(List* data, int row);
extern void PRINT_list_int(List* data, int row);

#endif