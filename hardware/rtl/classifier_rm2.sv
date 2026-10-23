// Reconfigurable Module Variant 2 (RM2): 2 Active Classes (Stage 1)
// Evaluates: Forest (0), River (1)
// Inactive: Highway (2) and Residential (3) clamped to -2^31

`timescale 1ns / 1ps

import classifier_rm_pkg::*;

module classifier_rm2 (
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                decouple,

    // Input Feature Stream
    input  logic                                feat_valid,
    input  logic [5:0]                          feat_idx,
    input  logic signed [DATA_WIDTH-1:0]        feat_data,
    output logic                                feat_ready,

    // Weight/Bias Configuration Interface
    input  logic                                cfg_we,
    input  logic [2:0]                          cfg_class,
    input  logic [5:0]                          cfg_addr,
    input  logic signed [DATA_WIDTH-1:0]        cfg_wdata,
    input  logic signed [SCORE_WIDTH-1:0]       cfg_bdata,

    // Fixed 3-Class Output Interface
    output logic                                score_valid,
    output logic signed [SCORE_WIDTH-1:0]       scores [0:MAX_CLASSES-1],
    output logic [MAX_CLASSES-1:0]              active_mask,
    output logic [2:0]                          predicted_class
);

    classifier_core #(
        .NUM_ACTIVE_CLASSES(2),
        .WEIGHT_MEM_FILE(""),
        .BIAS_MEM_FILE("")
    ) u_core (
        .clk(clk),
        .rst_n(rst_n),
        .decouple(decouple),
        .feat_valid(feat_valid),
        .feat_idx(feat_idx),
        .feat_data(feat_data),
        .feat_ready(feat_ready),
        .cfg_we(cfg_we),
        .cfg_class(cfg_class),
        .cfg_addr(cfg_addr),
        .cfg_wdata(cfg_wdata),
        .cfg_bdata(cfg_bdata),
        .score_valid(score_valid),
        .scores(scores),
        .active_mask(active_mask),
        .predicted_class(predicted_class)
    );

endmodule: classifier_rm2
