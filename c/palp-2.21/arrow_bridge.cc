#include "arrow_bridge.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static void print_schema_once(const std::shared_ptr<arrow::Schema>& schema) {
    static bool done = false;
    if (!done && schema) {
        std::cerr << "Schema: " << schema->ToString() << std::endl;
        done = true;
    }
}

static int process_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                         ks_row_callback cb, void* user_data)
{
    if (!batch) return 0;

    auto vc_col = batch->GetColumnByName("vertex_count");
    auto vertices_col = batch->GetColumnByName("vertices");

    if (!vc_col || !vertices_col) {
        std::cerr << "Missing required columns 'vertex_count' or 'vertices'\n";
        return false;
    }

    auto vc_array = std::dynamic_pointer_cast<arrow::Int32Array>(vc_col);
    if (!vc_array) {
        std::cerr << "'vertex_count' is not Int32: "
                  << vc_col->type()->ToString() << std::endl;
        return false;
    }

    auto outer = std::dynamic_pointer_cast<arrow::ListArray>(vertices_col);
    if (!outer) {
        std::cerr << "'vertices' is not outer ListArray: "
                  << vertices_col->type()->ToString() << std::endl;
        return false;
    }

    auto inner_values = outer->values();
    auto inner = std::dynamic_pointer_cast<arrow::ListArray>(inner_values);
    if (!inner) {
        std::cerr << "'vertices' values are not inner ListArray: "
                  << inner_values->type()->ToString() << std::endl;
        return false;
    }

    auto coord_values = std::dynamic_pointer_cast<arrow::Int32Array>(inner->values());
    if (!coord_values) {
        std::cerr << "Inner coordinate values are not Int32: "
                  << inner->values()->type()->ToString() << std::endl;
        return false;
    }

    const int64_t nrows = batch->num_rows();

    for (int64_t i = 0; i < nrows; i++) {
        if (vc_array->IsNull(i) || outer->IsNull(i)) continue;

        int vertex_count = static_cast<int>(vc_array->Value(i));
        if (vertex_count <= 0) continue;

        int64_t outer_start = outer->value_offset(i);
        int64_t outer_end   = outer->value_offset(i + 1);
        int64_t nverts      = outer_end - outer_start;

        if (nverts != vertex_count) {
            std::cerr << "Row mismatch: vertex_count=" << vertex_count
                      << " but vertices list has " << nverts << " vertices\n";
            continue;
        }

        long* coords = new long[vertex_count * 4];
        int idx = 0;
        bool bad = false;

        for (int64_t v = outer_start; v < outer_end; v++) {
            if (inner->IsNull(v)) {
                bad = true;
                break;
            }

            int64_t inner_start = inner->value_offset(v);
            int64_t inner_end   = inner->value_offset(v + 1);
            int64_t dim         = inner_end - inner_start;

            if (dim != 4) {
                std::cerr << "Expected 4 coordinates per vertex, got "
                          << dim << std::endl;
                bad = true;
                break;
            }

            for (int64_t j = inner_start; j < inner_end; j++) {
                coords[idx++] = static_cast<long>(coord_values->Value(j));
            }
        }

        if (bad) {
            delete[] coords;
            continue;
        }

        KSRow row;
        row.dim = 4;
        row.vertex_count = vertex_count;
        row.coords = coords;

        
        int rc = cb(&row, user_data);

        delete[] coords;

        if (rc > 0) {
            return 1;   // clean stop requested
        }
        if (rc < 0) {
            std::cerr << "Callback reported hard error\n";
            return -1;
        }
    }

    return 0;
}

extern "C" int ks_scan_arrow_dir(const char* dataset_dir,
                                 ks_row_callback cb,
                                 void* user_data)
{
    try {
        for (const auto& entry : fs::directory_iterator(dataset_dir)) {
            if (!entry.is_regular_file()) continue;

            std::string path = entry.path().string();
            if (path.find(".arrow") == std::string::npos) continue;

            std::cerr << "Reading file: " << path << std::endl;

            auto infile_result = arrow::io::ReadableFile::Open(path);
            if (!infile_result.ok()) {
                std::cerr << "Open failed: " << infile_result.status().ToString() << std::endl;
                continue;
            }
            std::shared_ptr<arrow::io::ReadableFile> infile = *infile_result;

            /* First try IPC file format */
            auto file_reader_result = arrow::ipc::RecordBatchFileReader::Open(infile);
            if (file_reader_result.ok()) {
                auto file_reader = *file_reader_result;
                print_schema_once(file_reader->schema());

                int num_batches = file_reader->num_record_batches();
                for (int b = 0; b < num_batches; b++) {
                    auto batch_result = file_reader->ReadRecordBatch(b);
                    if (!batch_result.ok()) {
                        std::cerr << "ReadRecordBatch(file) failed: "
                                  << batch_result.status().ToString() << std::endl;
                        break;
                    }
                    int prc = process_batch(*batch_result, cb, user_data);
                    if (prc > 0) return 0;   // clean stop
                    if (prc < 0) return 1;   // hard error
                }
                continue;
            }

            std::cerr << "FileReader failed, trying StreamReader: "
                      << file_reader_result.status().ToString() << std::endl;

            /* Re-open for stream mode */
            auto stream_result = arrow::io::ReadableFile::Open(path);
            if (!stream_result.ok()) {
                std::cerr << "Re-open for stream failed: "
                          << stream_result.status().ToString() << std::endl;
                continue;
            }
            std::shared_ptr<arrow::io::InputStream> stream = *stream_result;

            auto stream_reader_result = arrow::ipc::RecordBatchStreamReader::Open(stream);
            if (!stream_reader_result.ok()) {
                std::cerr << "StreamReader failed: "
                          << stream_reader_result.status().ToString() << std::endl;
                continue;
            }

            auto stream_reader = *stream_reader_result;
            print_schema_once(stream_reader->schema());

            while (true) {
                auto batch_result = stream_reader->Next();
                if (!batch_result.ok()) {
                    std::cerr << "Next(stream) failed: "
                              << batch_result.status().ToString() << std::endl;
                    break;
                }
                auto batch = *batch_result;
                if (!batch) break;

                int prc = process_batch(batch, cb, user_data);
                if (prc > 0) return 0;   // clean stop
                if (prc < 0) return 1;   // hard error
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}