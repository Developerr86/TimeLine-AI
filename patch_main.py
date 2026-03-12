import sys
import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add Response and stream_with_context to imports
if "Response, stream_with_context" not in content:
    content = content.replace(
        "from flask import Flask, jsonify, request, send_from_directory",
        "from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context"
    )

# 2. Patch analyze_contact_sheets
analyze_target = """    # Analyze each contact sheet
    all_selected_indices = []
    
    print(f"🔍 Analyzing {len(contact_sheets)} contact sheets...")
    
    def encode_image_to_base64(image_path):"""

if analyze_target in content:
    # Instead of replacing the whole thing, we know where it starts.
    # We will replace from `    # Analyze each contact sheet` up to the end of the function `    })`
    # Let's use regex to find the analyze_contact_sheets function body end.
    
    pattern = r"    # Analyze each contact sheet.*?    \}\)\n"
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        old_block = match.group(0)
        
        # New block:
        new_block = """    stream = request.args.get('stream', 'false').lower() == 'true'

    def encode_image_to_base64(image_path):
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    contact_sheet_prompt = \"\"\"
I am showing you 16 thumbnails from a lecture/video. 
Tell me the indices (1-16) of the frames that might contain text, graphs, diagrams, 
slides, code, or any other educationally useful information.
Ignore frames that show blank screens, loading screens, or non-educational content.

Respond with ONLY a valid JSON object containing an array of indices:
{"selected_indices": [3, 8, 12, 15]}

If NO frames appear to contain educational content, respond with:
{"selected_indices": []}
\"\"\"

    def process_generator():
        all_selected_indices = []
        if stream:
            yield f"data: {json.dumps({'type': 'status', 'message': f'Analyzing {len(contact_sheets)} contact sheets'})}\\n\\n"
            
        print(f"🔍 Analyzing {len(contact_sheets)} contact sheets...")
        
        for idx, contact_sheet_path in enumerate(contact_sheets):
            print(f"  📊 Analyzing contact sheet {idx + 1}/{len(contact_sheets)}...")
            if stream:
                img_url = f"/api/media/video/{session_id}/contact_sheets/{contact_sheet_path.name}"
                yield f"data: {json.dumps({'type': 'image', 'url': img_url})}\\n\\n"

            try:
                base64_image = encode_image_to_base64(contact_sheet_path)
                raw_response = ""
                
                if model_type == 'ollama':
                    import ollama
                    client = ollama.Client()
                    if stream:
                        response_stream = client.chat(
                            model=ollama_model,
                            messages=[
                                {
                                    'role': 'user',
                                    'content': contact_sheet_prompt,
                                    'images': [base64_image]
                                }
                            ],
                            options={"think": False},
                            stream=True
                        )
                        for chunk in response_stream:
                            token = chunk['message']['content']
                            raw_response += token
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\\n\\n"
                    else:
                        response = client.chat(
                            model=ollama_model,
                            messages=[
                                {
                                    'role': 'user',
                                    'content': contact_sheet_prompt,
                                    'images': [base64_image]
                                }
                            ],
                            options={"think": False}
                        )
                        raw_response = response['message']['content']
                    
                elif model_type == 'gemini':
                    import google.generativeai as genai
                    if not config.get('gemini_api_key'):
                        if stream:
                            yield f"data: {json.dumps({'type': 'error', 'message': 'Gemini API key not configured'})}\\n\\n"
                        continue
                    
                    genai.configure(api_key=config.get('gemini_api_key'))
                    model = genai.GenerativeModel(gemini_model)
                    with open(contact_sheet_path, 'rb') as image_file:
                        image_data = image_file.read()
                    
                    response = model.generate_content([contact_sheet_prompt, {"mime_type": "image/jpeg", "data": image_data}])
                    raw_response = response.text
                    
                elif model_type == 'remote':
                    import requests
                    with open(contact_sheet_path, 'rb') as img_file:
                        files = {'image': img_file}
                        data = {'prompt': contact_sheet_prompt}
                        response = requests.post(remote_url, files=files, data=data)
                        
                        if response.status_code == 200:
                            response_json = response.json()
                            raw_response = response_json.get('response', '')
                        else:
                            print(f"  ⚠️ Remote server error: {response.status_code}")
                            continue
                else:
                    continue
                
                # Parse JSON response
                if raw_response.strip().startswith("```json"):
                    json_str = raw_response.strip()[7:-3].strip()
                elif raw_response.strip().startswith("```"):
                    lines = raw_response.strip().split('\\n')
                    json_str = '\\n'.join(lines[1:-1]).strip()
                else:
                    json_str = raw_response
                
                try:
                    data = json.loads(json_str)
                    indices = data.get("selected_indices", [])
                    
                    # Convert 1-based indices to 0-based, and add offset for batch
                    batch_offset = idx * 16
                    extracted_count = 0
                    for i in indices:
                        if 1 <= i <= 16:
                            all_selected_indices.append(batch_offset + i)
                            extracted_count += 1
                    
                    print(f"    ✓ Selected indices from this sheet: {indices}")
                    if stream:
                         yield f"data: {json.dumps({'type': 'status', 'message': f'Extracted {extracted_count} frames'})}\\n\\n"
                    
                except json.JSONDecodeError:
                    print(f"  ⚠️ Failed to parse contact sheet response")
                    
            except Exception as e:
                print(f"  ⚠️ Error analyzing contact sheet {idx}: {e}")
                if stream:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
        
        # Handle case where no frames selected - fallback to all frames
        if not all_selected_indices:
            print("  ⚠️ No educational frames selected, falling back to all frames")
            all_selected_indices = list(range(1, total_frames + 1))
            use_fallback = True
        else:
            use_fallback = False
        
        print(f"  ✅ Total selected frames: {len(all_selected_indices)}")
        
        final_result = {
            'status': 'success',
            'selected_indices': sorted(all_selected_indices),
            'contact_sheets_created': len(contact_sheets),
            'contact_sheets_analyzed': len(contact_sheets),
            'total_frames': total_frames,
            'fallback_used': use_fallback
        }
        yield f"data: {json.dumps({'type': 'complete', 'result': final_result})}\\n\\n"

    if stream:
        return Response(stream_with_context(process_generator()), mimetype='text/event-stream')
    else:
        for chunk in process_generator():
            if chunk.startswith('data: '):
                try:
                    data = json.loads(chunk[6:].strip())
                    if data.get('type') == 'complete':
                        return jsonify(data['result'])
                    elif data.get('type') == 'error':
                        return jsonify({'status': 'error', 'message': data['message']}), 500
                except:
                    pass
        return jsonify({'status': 'error', 'message': 'Stream ended without completion'})
"""
        content = content.replace(old_block, new_block)
        print("Patched analyze_contact_sheets successfully.")
    else:
        print("Could not match analyze_contact_sheets")


# 3. Patch process_video_frames
if "def encode_image_to_base64(image_path):" in content:
    # Find the processing block
    pattern_frames = r"    def encode_image_to_base64\(image_path\):.*?    finally:\n        db\.close\(\)\n"
    match_frames = re.search(pattern_frames, content, flags=re.DOTALL)
    if match_frames:
        old_block_frames = match_frames.group(0)
        
        new_block_frames = """    stream = request.args.get('stream', 'false').lower() == 'true'
    
    def process_generator():
        def encode_image_to_base64(image_path):
            import base64
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        frame_prompt = \"\"\"
Analyze this video frame carefully and provide:
1. A brief description (1-2 sentences) of what this frame shows
2. Whether it contains text, diagrams, slides, code, or educational content

Respond with ONLY a valid JSON:
{"description": "Brief description of the frame content", "educational": "yes/no"}
\"\"\"
        
        descriptions = []
        frame_data_lines = []
        
        print(f"🖼️ Processing {len(frame_indices)} frames for session {session_id}")
        if stream:
            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {len(frame_indices)} frames'})}\\n\\n"
        
        # Create a mapping of frame index to file
        frame_file_map = {}
        for f in frame_files:
            idx = get_frame_index_from_filename(f.name)
            frame_file_map[idx] = f
        
        for i, frame_idx in enumerate(frame_indices):
            frame_file = frame_file_map.get(frame_idx)
            
            if not frame_file:
                print(f"  ⚠️ Frame {frame_idx} not found, skipping")
                continue
            
            timestamp = extract_timestamp_from_filename(frame_file.name)
            if not timestamp:
                timestamp = f"00:{frame_idx // 60:02d}:{frame_idx % 60:02d}"
            
            print(f"  📷 Processing frame {frame_idx} ({i+1}/{len(frame_indices)}) - {timestamp}")
            if stream:
                img_url = f"/api/media/video/{session_id}/frames/{frame_file.name}"
                yield f"data: {json.dumps({'type': 'image', 'url': img_url})}\\n\\n"
            
            try:
                base64_image = encode_image_to_base64(frame_file)
                raw_response = ""
                
                if model_type == 'ollama':
                    import ollama
                    client = ollama.Client()
                    if stream:
                        response_stream = client.chat(
                            model=ollama_model,
                            messages=[{'role': 'user', 'content': frame_prompt, 'images': [base64_image]}],
                            options={"think": False},
                            stream=True
                        )
                        for chunk in response_stream:
                            token = chunk['message']['content']
                            raw_response += token
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\\n\\n"
                    else:
                        response = client.chat(
                            model=ollama_model,
                            messages=[{'role': 'user', 'content': frame_prompt, 'images': [base64_image]}],
                            options={"think": False}
                        )
                        raw_response = response['message']['content']
                        
                elif model_type == 'gemini':
                    import google.generativeai as genai
                    if not config.get('gemini_api_key'):
                        if stream:
                            yield f"data: {json.dumps({'type': 'error', 'message': 'Gemini API key not configured'})}\\n\\n"
                        continue
                    
                    genai.configure(api_key=config.get('gemini_api_key'))
                    model = genai.GenerativeModel(gemini_model)
                    with open(frame_file, 'rb') as image_file:
                        image_data = image_file.read()
                    response = model.generate_content([frame_prompt, {"mime_type": "image/jpeg", "data": image_data}])
                    raw_response = response.text
                    
                elif model_type == 'remote':
                    import requests
                    with open(frame_file, 'rb') as img_file:
                        files = {'image': img_file}
                        data_p = {'prompt': frame_prompt}
                        response = requests.post(remote_url, files=files, data=data_p)
                        if response.status_code == 200:
                            response_json = response.json()
                            raw_response = response_json.get('response', '')
                        else:
                            continue
                else:
                    continue
                
                # Parse JSON
                if raw_response.strip().startswith("```json"):
                    json_str = raw_response.strip()[7:-3].strip()
                elif raw_response.strip().startswith("```"):
                    lines = raw_response.strip().split('\\n')
                    json_str = '\\n'.join(lines[1:-1]).strip()
                else:
                    json_str = raw_response
                
                try:
                    frame_data = json.loads(json_str)
                    description = frame_data.get("description", "No description")
                    descriptions.append({"frame": frame_idx, "timestamp": timestamp, "description": description})
                    frame_data_lines.append(f"[{timestamp}] Frame {frame_idx}: {description}")
                except json.JSONDecodeError:
                    descriptions.append({"frame": frame_idx, "timestamp": timestamp, "description": raw_response[:200] if raw_response else "Failed to parse"})
                    frame_data_lines.append(f"[{timestamp}] Frame {frame_idx}: {raw_response[:200]}")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing frame {frame_idx}: {e}")
                if stream:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
        
        # Write frame data to file
        try:
            with open(frame_data_file, 'w', encoding='utf-8') as f:
                f.write("=== Video Frame Analysis ===\\n\\n")
                f.write("\\n".join(frame_data_lines))
            print(f"  ✅ Frame data saved to: {frame_data_file}")
            
            db = get_db()
            try:
                frame_data_text = "\\n".join(frame_data_lines)
                text_record = CapturedText(
                    session_id=session_id,
                    content=frame_data_text,
                    content_type="frame_data"
                )
                db.add(text_record)
                db.commit()
            except Exception as e:
                print(f"  ⚠️ Failed to save frame data to database: {e}")
            finally:
                db.close()
                
        except Exception as e:
            print(f"  ⚠️ Failed to save frame data: {e}")
            
        final_result = {
            'status': 'success',
            'frames_processed': len(descriptions),
            'frame_data_file': "frame_data.txt",
            'descriptions': descriptions
        }
        yield f"data: {json.dumps({'type': 'complete', 'result': final_result})}\\n\\n"

    if stream:
        return Response(stream_with_context(process_generator()), mimetype='text/event-stream')
    else:
        for chunk in process_generator():
            if chunk.startswith('data: '):
                try:
                    data = json.loads(chunk[6:].strip())
                    if data.get('type') == 'complete':
                        return jsonify(data['result'])
                    elif data.get('type') == 'error':
                        return jsonify({'status': 'error', 'message': data['message']}), 500
                except:
                    pass
        return jsonify({'status': 'error', 'message': 'Stream ended without completion'})
"""
        content = content.replace(old_block_frames, new_block_frames)
        print("Patched process_video_frames successfully.")
    else:
        print("Could not match process_video_frames")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done.")
