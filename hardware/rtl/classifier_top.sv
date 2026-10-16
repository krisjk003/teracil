// Static System Top-Level Wrapper Hosting the Reconfigurable Partition (RP)
// Provides DFX decoupling control, interface handshaking, and status telemetry.

`timescale 1ns / 1ps

import classifier_rm_pkg::*;

module classifier_top (
    input  logic                                clk,
    input  logic                                rst_n,

    // Dynamic Function eXchange (DFX) Controller Interface
    input  logic                                dfx_decouple,        // Asserted high during partial reconfiguration
    output logic                                dfx_ready,           // Static region ready status

    // Input Feature Stream (from static CNN feature extractor)
    input  logic                                feat_valid,
    input  logic [5:0]                          feat_idx,
    input  logic signed [DATA_WIDTH-1:0]        feat_data,
    output logic                                feat_ready,

    // Reconfigurable Module Programming Bus
    input  logic                                cfg_we,
    input  logic [2:0]                          cfg_class,
    input  logic [5:0]                          cfg_addr,
    input  logic signed [DATA_WIDTH-1:0]        cfg_wdata,
    input  logic signed [SCORE_WIDTH-1:0]       cfg_bdata,

    // System Classification Output
    output logic                                score_valid,
    output logic signed [SCORE_WIDTH-1:0]       scores [0:MAX_CLASSES-1],
    output logic [MAX_CLASSES-1:0]              active_mask,
    output logic [2:0]                          predicted_class
);

    // Decoupling latch to prevent glitches during reconfigurable partition swap
    logic decouple_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            decouple_reg <= 1'b1;
        end else begin
            decouple_reg <= dfx_decouple;
        end
    end

    assign dfx_ready = !decouple_reg;

    // Reconfigurable Partition Instantiation
    // In Vivado DFX, this hierarchical boundary is marked as HD.RECONFIGURABLE
    classifier_rm3 u_reconfigurable_partition (
        .clk(clk),
        .rst_n(rst_n),
        .decouple(decouple_reg),
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

endmodule: classifier_top
