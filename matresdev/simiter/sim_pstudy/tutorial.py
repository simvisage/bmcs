#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Feb 3, 2010 by: rch

if __name__ == '__main__':

    # DEFINE A SIMULATION MODEL
    # -------------------------
    # import a model - a class that provides
    # - factors         as traits with a prescribed set of levels
    #                   that should be included in the parametric study.
    #                   A trait is regarded as Factor if it specifies
    #                   a set of levels in form ps_levels = True.
    # - get_sim_outputs()   as instances of SimOut class
    #                   defining the name and order of outputs
    # - peval()          method returning a vector of results
    #                   the order of outputs is prescribed by the
    #                   specification given in the get_outputs() method
    #
    # Here we just import predefined Foo model with three four inputs
    #
    # [ index_1, material_model, param_1, param_2 ]
    #
    # The inputs have the types [ Int, Callable, Float, Float
    #
    from .sim_model import SimModel
    sim_model = SimModel()

    # The model response is obtained by issuing
    #
    print('default evaluation', sim_model.peval())

    # returning an array with two values.
    # In this call, the default values of the factors
    # [input_1, material_model, param_1, param_2 ]
    # were taken. The factor levels were ignored.

    # DEFINING A STUDY
    # --------------
    # In order to study the response of the model in a broader range
    # of specified levels we now construct the parametric study
    #
    from .sim_pstudy import SimArray
    pstudy = SimArray(sim_model=sim_model)

    # ACCESSING OUTPUTS
    # --------------
    # The pstudy can be regarded as an n-dimensional array
    # providing the results for each combination of factor levels
    # In order to get the model response for the first level
    # of all parameters the index operator can be used as follows:
    #
    print('model output for ground levels', pstudy[0, 0, 0, 0])

    # The combination of last levels is obtained as
    #
    print('model output for floor levels', pstudy[-1, -1, -1, -1])

    # The computation of outputs is performed on demand and cached.
    # Thus, if an index appears the second time only the cached value
    # is returned.
    #
    print('lookup the output in the cache', pstudy[-1, -1, -1, -1])

    # Just like for any array, indexes may be sliced
    #
    print('get the outputs for all levels of index_1', pstudy[:, 0, 0, 0])

    # the result of this call is 2-dimensional array with the first index
    # specifying the level of index_1 and second index giving the output

    # In analogy, for slice over the last two indexes
    #
    print('get the outputs for all combinations of param_1 x param_2', pstudy[0, 0, :, :])

    # a 3-dimensional array is returned with first two indexes correspond
    # to the levels of param_1 and param_2 and third index identifying the
    # output

    # Finally, the whole study can be performed using ellipsis
    #
    print('get all the values in the n-dimensional space', pstudy[...])

    # The result is a 5-dimensional array with first four indexes
    # denoting the factor levels and last index the output.
    # Note that the previously accessed slices are reused in this call.

    # MODIFYING LEVELS
    # ----------------
    # The initially specified set of default levels for each factor
    # can be changed within an existing study.
    #
    pstudy.factor_dict['param_1'].max_level = 10
    pstudy.factor_dict['param_1'].n_levels = 5
    print('get output for first two levels of param_1', pstudy[-1, -1, 1, -1])

    # Note that values for pstudy[:,:,0,:] were included in the old
    # grid of levels and are reused in the new study as well.

    # COMMENTS
    # --------
    # The study is limited to an a regular grid of levels. This can be regarded
    # as a full factorial in the nomenclature of the Design of Experiments (DOE).
    # The table showing the full factorial in the user interface can be seen by issuing
    #
    pstudy.configure_traits()

    # The evaluation of the model is not performed for all possible
    # combinations of factor levels. The evaluation is done first when
    # the particular value is accessed using the level indexes.
    #
    # The numpy indexing functions are used to access values in the
    # factor space. The data structure can be used for
    # - on demand construction of 2D and 3D views into the output
    #   space of the study
    # - construction of a regression model within the factor space in analogy
    #   to the DOE
    # - supporting adaptive integration procedures within the factor space
    #   as it is the case in statistical analysis  of multi-variate response.
    #
    # An example of the first application can be demonstrated by using the
    # SimArrayView class
    #
    from .sim_array_view import SimArrayView
    SimArrayView(model=pstudy).configure_traits()

    # SAVING THE STUDY
    # ----------------
    # In order to save the study, it the SimPStudy class manages
    # the object persistence. It associates the study with a file name
    # and monitors whether the study has been changed or not.
    #
    from .sim_pstudy import SimPStudy
    sim_pstudy = SimPStudy(sim_model=sim_model)
    sim_pstudy.configure_traits()

    # RESETTING THE CACHE
    # -------------------

    # LOADING AN EXISTING STUDY
    # -------------------------
