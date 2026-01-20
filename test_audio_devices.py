import soundcard as sc

print("🔍 Diagnostics: Audio Device Detection")
print("=" * 40)

try:
    # 1. List all speakers
    print("\n🔊 Output Devices (Speakers):")
    speakers = sc.all_speakers()
    for s in speakers:
        print(f"  - {s.name}")

    # 2. List all microphones (including loopback)
    print("\n🎤 Input Devices (Microphones):")
    mics = sc.all_microphones(include_loopback=True)
    stereo_mix_found = False
    
    for m in mics:
        loopback_tag = " [Loopback]" if "loopback" in m.name.lower() or "stereo mix" in m.name.lower() else "" 
        # Note: Stereo Mix is technically an input, not a loopback output virtual wrapper, but acts similarly for recording
        
        print(f"  - {m.name}{loopback_tag}")
        
        if "stereo mix" in m.name.lower():
            stereo_mix_found = True

    print("\n" + "=" * 40)
    if stereo_mix_found:
        print("✅ SUCCESS: 'Stereo Mix' device FOUND!")
        print("   The new code will select this device for recording.")
    else:
        print("⚠️ WARNING: 'Stereo Mix' device NOT FOUND.")
        print("   The new code will fallback to default loopback.")
        print("   To enable Stereo Mix: Open Sound Settings > Manage Audio Devices > Recording > Enable Stereo Mix")

except Exception as e:
    print(f"❌ Error during detection: {e}")
