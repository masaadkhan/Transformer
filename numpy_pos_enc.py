# int token_idx = row;
# int current_dim = col;
# int token_dims = num_cols;

# pos_enc[idx] = (current_dim & 1) ?
#               sinf(token_idx) / powf(10000, (2 * current_dim) / token_dims) :
#               cosf(token_idx) / powf(10000, (2 * current_dim) / token_dims);

rows_np = np.arange(pos_enc_sequence_len)[:, np.newaxis]
even_rows = (rows_np % 2) == 0
odd_rows = ~even_rows

# pos_encodings_np_T[even_rows] = np.sin(pos_encodings_np_T[even_rows])
# pos_encodings_np_T[odd_rows] = np.cos(pos_encodings_np_T[even_rows])

pos_cols_np = np.arange(token_dims)
pos_cols_np = 10000 ** ((2 * pos_cols_np) / token_dims)

# angle_rads = pos_encodings_np_T * pos_cols
# print(pos_encodings_np_T)
# print(pos_cols)
# print(angle_rads)

rows_np[even_rows] = np.sin(rows_np[even_rows] / pos_cols_np)
rows_np[odd_rows] = np.cos(rows_np[odd_rows] / pos_cols_np)

print(rows_np)


# print(even_rows)