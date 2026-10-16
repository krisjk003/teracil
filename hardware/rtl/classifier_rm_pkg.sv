// Package defining parameters, types, and fixed interface specifications
// for EuroSAT Class-Incremental Reconfigurable Modules (RM3, RM4, RM5).

`timescale 1ns / 1ps

package classifier_rm_pkg;

    localparam int FEATURE_DIM       = 64;
    localparam int DATA_WIDTH        = 8;
    localparam int SCORE_WIDTH       = 32;
    localparam int MAX_CLASSES       = 5;
    localparam logic signed [SCORE_WIDTH-1:0] INACTIVE_SCORE = 32'sh8000_0000; // -2,147,483,648

    // Class Index Enumeration strictly matching Python training order
    typedef enum logic [2:0] {
        CLS_FOREST      = 3'd0,
        CLS_RESIDENTIAL = 3'd1,
        CLS_RIVER       = 3'd2,
        CLS_HIGHWAY     = 3'd3,
        CLS_INDUSTRIAL  = 3'd4,
        CLS_INVALID     = 3'd7
    } class_id_t;

endpackage: classifier_rm_pkg
