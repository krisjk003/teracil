// Synthesizable Classifier Core with Dynamic Reconfigurable Partition Support
// Supports on-the-fly streaming MAC accumulation and fixed 5-class output interface.

`timescale 1ns / 1ps

import classifier_rm_pkg::*;

module classifier_core #(
    parameter int NUM_ACTIVE_CLASSES = 3,
    parameter string WEIGHT_MEM_FILE = "",
    parameter string BIAS_MEM_FILE   = ""
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                decouple,    // Active-high DFX decoupling signal

    // Input Feature Stream (64 elements, INT8)
    input  logic                                feat_valid,
    input  logic [5:0]                          feat_idx,
    input  logic signed [DATA_WIDTH-1:0]        feat_data,
    output logic                                feat_ready,

    // Weight & Bias Programming / Initialization Interface
    input  logic                                cfg_we,
    input  logic [2:0]                          cfg_class,
    input  logic [5:0]                          cfg_addr,
    input  logic signed [DATA_WIDTH-1:0]        cfg_wdata,
    input  logic signed [SCORE_WIDTH-1:0]       cfg_bdata,

    // Fixed 5-Class Output Interface
    output logic                                score_valid,
    output logic signed [SCORE_WIDTH-1:0]       scores [0:MAX_CLASSES-1],
    output logic [MAX_CLASSES-1:0]              active_mask,
    output logic [2:0]                          predicted_class
);

    // On-chip storage for weights and biases of active classes
    logic signed [DATA_WIDTH-1:0]  weights [0:NUM_ACTIVE_CLASSES-1][0:FEATURE_DIM-1];
    logic signed [SCORE_WIDTH-1:0] biases  [0:NUM_ACTIVE_CLASSES-1];

    // Optional memory file preloading
    initial begin
        if (WEIGHT_MEM_FILE != "") begin
            $readmemh(WEIGHT_MEM_FILE, weights);
        end
        if (BIAS_MEM_FILE != "") begin
            $readmemh(BIAS_MEM_FILE, biases);
        end
    end

    // Programmable weight/bias update logic
    always_ff @(posedge clk) begin
        if (cfg_we && (cfg_class < NUM_ACTIVE_CLASSES)) begin
            weights[cfg_class][cfg_addr] <= cfg_wdata;
            if (cfg_addr == 6'd0) begin
                biases[cfg_class] <= cfg_bdata;
            end
        end
    end

    // On-the-fly accumulation registers
    logic signed [SCORE_WIDTH-1:0] accum [0:NUM_ACTIVE_CLASSES-1];
    logic signed [SCORE_WIDTH-1:0] out_scores [0:MAX_CLASSES-1];
    logic                          out_valid;

    // Feature processing and dot-product accumulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            for (int c = 0; c < NUM_ACTIVE_CLASSES; c++) begin
                accum[c] <= '0;
            end
        end else if (decouple) begin
            out_valid <= 1'b0;
            for (int c = 0; c < NUM_ACTIVE_CLASSES; c++) begin
                accum[c] <= '0;
            end
        end else begin
            out_valid <= 1'b0;
            if (feat_valid) begin
                for (int c = 0; c < NUM_ACTIVE_CLASSES; c++) begin
                    automatic logic signed [15:0] prod = feat_data * weights[c][feat_idx];
                    if (feat_idx == 6'd0) begin
                        // Start of new vector: initialize with bias + first product
                        accum[c] <= biases[c] + prod;
                    end else begin
                        accum[c] <= accum[c] + prod;
                    end
                end

                // Completion of 64-element feature vector
                if (feat_idx == 6'd63) begin
                    out_valid <= 1'b1;
                end
            end
        end
    end

    // Latch final scores and valid flag upon completion
    logic score_valid_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score_valid_reg <= 1'b0;
            for (int i = 0; i < MAX_CLASSES; i++) begin
                out_scores[i] <= INACTIVE_SCORE;
            end
        end else if (decouple) begin
            score_valid_reg <= 1'b0;
            for (int i = 0; i < MAX_CLASSES; i++) begin
                out_scores[i] <= INACTIVE_SCORE;
            end
        end else if (out_valid) begin
            score_valid_reg <= 1'b1;
            for (int i = 0; i < MAX_CLASSES; i++) begin
                if (i < NUM_ACTIVE_CLASSES) begin
                    out_scores[i] <= accum[i];
                end else begin
                    // Guarantee inactive classes can never win argmax
                    out_scores[i] <= INACTIVE_SCORE;
                end
            end
        end else if (feat_valid && feat_idx == 6'd0) begin
            // Clear valid flag when a new vector starts streaming
            score_valid_reg <= 1'b0;
        end
    end

    // Argmax Decision Logic strictly over active classes
    logic [2:0] argmax_class;
    always_comb begin
        argmax_class = 3'd0;
        for (int c = 1; c < NUM_ACTIVE_CLASSES; c++) begin
            if (out_scores[c] > out_scores[argmax_class]) begin
                argmax_class = c[2:0];
            end
        end
    end

    // Output assignments with DFX decoupling protection
    assign feat_ready = !decouple;
    assign score_valid = decouple ? 1'b0 : score_valid_reg;
    assign predicted_class = decouple ? CLS_INVALID : argmax_class;

    always_comb begin
        for (int i = 0; i < MAX_CLASSES; i++) begin
            scores[i] = decouple ? INACTIVE_SCORE : out_scores[i];
            active_mask[i] = decouple ? 1'b0 : (i < NUM_ACTIVE_CLASSES);
        end
    end

endmodule: classifier_core
