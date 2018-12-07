
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from ibvpy.api import FEDomain, FEGrid, FERefinementGrid, TStepper as TS
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q

if __name__ == '__main__':

    fets_eval_4u = FETS2D4Q(mats_eval = MATS2DElastic())
    
    fe_domain = FEDomain()

    fe_rgrid1 = FERefinementGrid( name = 'fe_rgrid1', fets_eval = fets_eval_4u, domain = fe_domain )

    fe_grid1 = FEGrid( name = 'fe_grid1', coord_max = (2.,6.,0.), 
                               shape   = (1,3),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid1 )    

    fe_grid2 = FEGrid( name = 'fe_grid2', coord_min = (2.,  6, 0.),
                      coord_max = (10, 15, 0.), 
                               shape   = (3,2),
                               fets_eval = fets_eval_4u,
                               level = fe_rgrid1 )    
    
    print(fe_grid2[ 1, 0 ].elems)
    
    fe_grid2.deactivate( ( 1, 0 ) )
    print('activation map')
    print(fe_grid2.activation_map)

    ts = TS( sdomain = fe_domain )
    
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = ts )
    ibvpy_app.main()    