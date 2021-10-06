\ Model mcfp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x_e[0] + 5 x_e[1] + x_e[2] + x_e[3] + 6 x_e[4] + x_e[5] + x_e[6]
Subject To
 demand_constraints[0]: f_kj[0,0] + f_kj[0,1] = 5
 demand_constraints[1]: f_kj[1,0] + f_kj[1,1] = 3
 def_constraints[0]: f_kj[0,1] - x_e[0] = 0
 def_constraints[1]: f_kj[0,0] - x_e[1] = 0
 def_constraints[2]: f_kj[0,1] + f_kj[1,1] - x_e[2] = 0
 def_constraints[3]: f_kj[1,1] - x_e[3] = 0
 def_constraints[4]: f_kj[1,0] - x_e[4] = 0
 def_constraints[5]: f_kj[0,1] - x_e[5] = 0
 def_constraints[6]: f_kj[1,1] - x_e[6] = 0
 edge_capacity_constraints[2]: x_e[2] <= 5
 edge_capacity_constraints[3]: x_e[3] <= 2
 edge_capacity_constraints_1[0,0]: - 9999 f_kj[0,0] + del_f_kj[0,0] <= 0
 edge_capacity_constraints_1[0,1]: - 9999 f_kj[0,1] + del_f_kj[0,1] <= 0
 edge_capacity_constraints_1[1,0]: - 9999 f_kj[1,0] + del_f_kj[1,0] <= 0
 edge_capacity_constraints_1[1,1]: - 9999 f_kj[1,1] + del_f_kj[1,1] <= 0
 edge_capacity_constraints_2[0,0]: f_kj[0,0] - 9999 del_f_kj[0,0] <= 0
 edge_capacity_constraints_2[0,1]: f_kj[0,1] - 9999 del_f_kj[0,1] <= 0
 edge_capacity_constraints_2[1,0]: f_kj[1,0] - 9999 del_f_kj[1,0] <= 0
 edge_capacity_constraints_2[1,1]: f_kj[1,1] - 9999 del_f_kj[1,1] <= 0
 edge_capacity_constraints_no_paths[0]: del_f_kj[0,0] + del_f_kj[0,1] = 1
 edge_capacity_constraints_no_paths[1]: del_f_kj[1,0] + del_f_kj[1,1] = 1
Bounds
Binaries
 del_f_kj[0,0] del_f_kj[0,1] del_f_kj[1,0] del_f_kj[1,1]
End
