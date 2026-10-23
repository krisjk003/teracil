# Vivado SystemVerilog Simulation Script
# Compiles RTL, testbench, and executes simulation batch in Vivado xsim

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set PROJ_ROOT  [file normalize "$SCRIPT_DIR/../.."]
set RTL_DIR    "$PROJ_ROOT/hardware/rtl"
set TB_DIR     "$PROJ_ROOT/hardware/testbench"
set WORK_DIR   "$PROJ_ROOT/results/sim_work"

file mkdir $WORK_DIR
cd $WORK_DIR

# Compile SystemVerilog files
exec xvlog -sv "$RTL_DIR/classifier_rm_pkg.sv" "$RTL_DIR/classifier_core.sv" "$RTL_DIR/classifier_rm2.sv" "$RTL_DIR/classifier_rm3.sv" "$RTL_DIR/classifier_rm4.sv" "$TB_DIR/tb_classifier_rm.sv"


# Elaborate
exec xelab tb_classifier_rm -s tb_classifier_rm_sim -debug typical

# Run simulation
exec xsim tb_classifier_rm_sim -R

puts "Simulation Completed Successfully."
