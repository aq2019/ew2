import pandas as pd
import s3fs
import sys
!conda install --yes --prefix {sys.prefix} pyarrow

conversations_path = 's3://ew-insight-2020b-bucket/ewdb-export-june3-2020-second-attempt/ewdb/public.conversations/part-00000-f2d72c97-65c1-4cf9-84f8-41c5960cdabd-c000.gz.parquet'
conversations_tag_path = 's3://ew-insight-2020b-bucket/ewdb-export-june3-2020-second-attempt/ewdb/public.conversations_tag/part-00000-c0778d74-f0f9-45cf-9b47-a467bc9ee925-c000.gz.parquet'
messages_path = 's3://ew-insight-2020b-bucket/ewdb-export-june3-2020-second-attempt/ewdb/public.messages/part-00000-b0254068-39b1-4ac2-8a21-c0b45be7110f-c000.gz.parquet'
tags_dictionary_path = 's3://ew-insight-2020b-bucket/ewdb-export-june3-2020-second-attempt/ewdb/public.tags_dictionary/part-00000-cd79a5bf-2d01-400f-88a2-21e45de67522-c000.gz.parquet'

conv_df = pd.read_parquet(conversations_path, engine='pyarrow')
conv_tag_df = pd.read_parquet(conversations_tag_path, engine = 'pyarrow')
messages_df = pd.read_parquet(messages_path, engine = 'pyarrow')
tags_dict_df = pd.read_parquet(tags_dictionary_path, engine = 'pyarrow') 

conv_tag_df.to_csv('conv_tag.csv', index=False)
conv_tag_df.to_pickle('conv_tag.pkl')

conv_df.to_csv('conv.csv', index=False)
conv_df.to_pickle('conv.pkl')

messages_df.to_csv('messages.csv', index=False)
messages_df.to_pickle('messages.pkl')

tags_dict_df.to_csv('tags_dict.csv', index=False)
tags_dict_df.to_pickle('tags_dict.pkl')

