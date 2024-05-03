import os
import asyncio
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import sqlite3
import pickle

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Directory containing image files')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def prepare_dtb():
    conn = sqlite3.connect('./db.sqlite3')
    _cur = conn.cursor()
    # Create table if not exists
    _cur.execute('''CREATE TABLE IF NOT EXISTS product_data (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, data BINARY NOT NULL, url TEXT)''')
    return _cur, conn

def clean_results(results):
    cleaned_results = []
    title = ''
    for result in results:
        category, _ = result[0].split('|')
        attributes = result[2].replace('\n', ', ')
        color = result[3].capitalize()
        first_attribute_name = attributes.split(',')[0].strip().capitalize()
        if title == '':
            title = f"{category.capitalize()} {first_attribute_name} - {color}"
        cleaned_results.append((category, result[1], result[2], result[3]))
    return title, cleaned_results

def add_to_db(_conn, _cur, _res, _url):
    _ttl, _det = clean_results(_res)
    pickled_data = pickle.dumps(_det, protocol=pickle.HIGHEST_PROTOCOL)
    _cur.execute(f"INSERT INTO product_data (title, data, url) VALUES (?, ?, ?) ON CONFLICT DO UPDATE SET data=data", (_ttl, pickled_data, _url),
    )
    _conn.commit()
    return f"Done insert for {_url}"

def main(args):
    _cur, _conn = prepare_dtb()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    files = os.listdir(args.img_dir)
    image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for img_file in image_files:
        img_path = os.path.join(args.img_dir, img_file)
        result = inference_detector(model, img_path)
        res_img, res_det = model._show_result_cv(img_path, result, score_thr=args.score_thr, out_file=f'outputx/demo_{img_file}')
        print(f"Done detection for {img_file}")
        if res_det is None or res_det == []:
            print(f"No object found in file {img_file}")
        else:
            resd = add_to_db(_conn, _cur, res_det, img_file)
            print(resd)

    _conn.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)