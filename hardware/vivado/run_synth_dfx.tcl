# Vivado Synthesis and Dynamic Function eXchange (DFX) Implementation Flow
# EuroSAT Class-Incremental Classifier Prototype
# Target: Configurable FPGA (Default: Xilinx Zynq-7000 xc7z020clg400-1)

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set PROJ_ROOT  [file normalize "$SCRIPT_DIR/../.."]
set RTL_DIR    "$PROJ_ROOT/hardware/rtl"
set RPT_DIR    "$PROJ_ROOT/results/hardware_reports"

file mkdir $RPT_DIR

# Configuration parameters (can be overridden via environment or command line args)
if {[info exists env(FPGA_PART)]} {
    set TARGET_PART $env(FPGA_PART)
} else {
    set TARGET_PART "xc7z020clg400-1"
}

puts "=========================================================================="
puts "Target FPGA Part: $TARGET_PART"
puts "Running Out-of-Context (OOC) Synthesis for Reconfigurable Modules (RMs)"
puts "=========================================================================="

# List of reconfigurable modules
set RM_LIST [list "classifier_rm3" "classifier_rm4" "classifier_rm5"]

foreach rm $RM_LIST {
    puts "\n--------------------------------------------------"
    puts "Synthesizing Reconfigurable Module: $rm"
    puts "--------------------------------------------------"
    
    # Create an in-memory project for clean OOC synthesis
    create_project -in_memory -part $TARGET_PART
    
    # Read SystemVerilog files
    read_verilog -sv "$RTL_DIR/classifier_rm_pkg.sv"
    read_verilog -sv "$RTL_DIR/classifier_core.sv"
    read_verilog -sv "$RTL_DIR/${rm}.sv"
    
    # Out-of-Context synthesis for reconfigurable partition
    synth_design -top $rm -part $TARGET_PART -mode out_of_context
    
    # Generate utilization report
    report_utilization -file "$RPT_DIR/${rm}_utilization.rpt"
    report_timing_summary -file "$RPT_DIR/${rm}_timing.rpt"
    
    # Write checkpoint
    write_checkpoint -force "$RPT_DIR/${rm}_synth.dcp"
    
    close_project
}

puts "\n--------------------------------------------------"
puts "Synthesizing Static Top with Reconfigurable Partition"
puts "--------------------------------------------------"
create_project -in_memory -part $TARGET_PART
read_verilog -sv "$RTL_DIR/classifier_rm_pkg.sv"
read_verilog -sv "$RTL_DIR/classifier_core.sv"
read_verilog -sv "$RTL_DIR/classifier_rm3.sv"
read_verilog -sv "$RTL_DIR/classifier_top.sv"

synth_design -top classifier_top -part $TARGET_PART
report_utilization -file "$RPT_DIR/static_top_utilization.rpt"
write_checkpoint -force "$RPT_DIR/static_top_synth.dcp"
close_project

puts "=========================================================================="
puts "Synthesis and Reports Generated Successfully in: $RPT_DIR"
puts "=========================================================================="
