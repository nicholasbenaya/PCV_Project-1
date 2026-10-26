"""Sistem audio prosedural (SFX dan BGM adaptif berbasis intensitas permainan).

Menghasilkan gelombang suara chiptune arcade secara sintetis menggunakan NumPy
dan memutarnya melalui pygame.mixer tanpa memerlukan file audio eksternal.
"""

from typing import Dict, Optional
import numpy as np

try:
    import pygame
    PYGAME_AUDIO_AVAILABLE = True
except ImportError:
    PYGAME_AUDIO_AVAILABLE = False


class AudioSystem:
    """Manajer efek suara (SFX) dan musik latar belakang (BGM) dinamis."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.enabled = True
        self.sfx_volume = 0.8
        self.bgm_volume = 0.4
        self.current_intensity = 0
        self.active_bgm_channel: Optional[object] = None

        self.sfx_sounds: Dict[str, object] = {}
        self.bgm_loops: Dict[int, object] = {}

        if PYGAME_AUDIO_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(
                        frequency=self.sample_rate,
                        size=-16,
                        channels=2,
                        buffer=512,
                    )
                self._generate_sfx()
                self._generate_bgm_loops()
            except Exception:
                self.enabled = False
        else:
            self.enabled = False

    def _create_stereo_sound(self, mono_samples: np.ndarray) -> Optional[object]:
        """Mengonversi sinyal mono float (-1.0 s.d 1.0) menjadi pygame.mixer.Sound stereo."""
        if not PYGAME_AUDIO_AVAILABLE or not self.enabled:
            return None
        clamped = np.clip(mono_samples, -1.0, 1.0)
        pcm = (clamped * 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack((pcm, pcm)))
        return pygame.sndarray.make_sound(stereo)

    def _generate_sfx(self) -> None:
        """Menghasilkan seluruh sampel efek suara arcade secara prosedural."""
        sr = self.sample_rate

        # SFX 1: Click / Navigasi Menu (Pip nada tinggi)
        dur = 0.05
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        wave_click = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 60)
        snd_click = self._create_stereo_sound(wave_click * 0.6)
        if snd_click:
            self.sfx_sounds["click"] = snd_click

        # SFX 2: Whack / Bonk (Pukulan kena mole - sapuan frekuensi turun cepat)
        dur = 0.12
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = np.linspace(440, 90, len(t))
        wave_bonk = (
            np.sin(2 * np.pi * freq * t) * 0.7
            + np.sin(4 * np.pi * freq * t) * 0.3
        ) * np.exp(-t * 22)
        snd_bonk = self._create_stereo_sound(wave_bonk * 0.9)
        if snd_bonk:
            self.sfx_sounds["whack"] = snd_bonk

        # SFX 3: Miss / Mole Kabur (Nada dengung rendah)
        dur = 0.18
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = np.linspace(190, 80, len(t))
        wave_miss = np.sign(np.sin(2 * np.pi * freq * t)) * np.exp(-t * 12) * 0.4
        snd_miss = self._create_stereo_sound(wave_miss)
        if snd_miss:
            self.sfx_sounds["miss"] = snd_miss

        # SFX 4: Level Clear Fanfare (Arpeggio akor C mayor: C5 - E5 - G5 - C6)
        notes = [523.25, 659.25, 783.99, 1046.50]
        chord_parts = []
        part_dur = 0.10
        t_part = np.linspace(0, part_dur, int(sr * part_dur), endpoint=False)
        for freq in notes:
            wave = (
                np.sin(2 * np.pi * freq * t_part) * 0.6
                + np.sin(4 * np.pi * freq * t_part) * 0.2
            ) * np.exp(-t_part * 8)
            chord_parts.append(wave)
        chord_wave = np.concatenate(chord_parts)
        snd_clear = self._create_stereo_sound(chord_wave * 0.75)
        if snd_clear:
            self.sfx_sounds["level_clear"] = snd_clear

        # SFX 5: Game Over (Arpeggio nada minor turun)
        notes_go = [440.0, 369.99, 311.13, 220.0]
        go_parts = []
        part_dur = 0.14
        t_part = np.linspace(0, part_dur, int(sr * part_dur), endpoint=False)
        for freq in notes_go:
            wave = np.sin(2 * np.pi * freq * t_part) * np.exp(-t_part * 10)
            go_parts.append(wave)
        go_wave = np.concatenate(go_parts)
        snd_go = self._create_stereo_sound(go_wave * 0.8)
        if snd_go:
            self.sfx_sounds["game_over"] = snd_go

    def _generate_bgm_loops(self) -> None:
        """Menghasilkan 4 tingkat loop musik sintetis berirama dengan intensitas bertahap."""
        sr = self.sample_rate

        # Intensitas 1: Santai (Tempo ~110 BPM, pola bass sederhana)
        bpm1 = 110
        beat_len1 = int(sr * (60.0 / bpm1) / 2)  # Eighth note
        bass_seq1 = [130.81, 130.81, 164.81, 146.83]  # C3, C3, E3, D3
        samples1 = []
        for f in bass_seq1 * 2:
            t = np.linspace(0, beat_len1 / sr, beat_len1, endpoint=False)
            wave = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 7)
            samples1.append(wave)
        loop1 = self._create_stereo_sound(np.concatenate(samples1) * 0.28)
        if loop1:
            self.bgm_loops[1] = loop1

        # Intensitas 2: Aktif (Tempo ~130 BPM, bass berjalan dan pulsa snare lembut)
        bpm2 = 130
        beat_len2 = int(sr * (60.0 / bpm2) / 2)
        bass_seq2 = [130.81, 196.00, 164.81, 220.00]
        samples2 = []
        for i, f in enumerate(bass_seq2 * 2):
            t = np.linspace(0, beat_len2 / sr, beat_len2, endpoint=False)
            synth = np.sin(2 * np.pi * f * t) * np.exp(-t * 8)
            # Ketukan drum setiap ketukan genap
            snare = (np.random.uniform(-0.15, 0.15, len(t)) * np.exp(-t * 25)) if (i % 2 == 1) else 0
            samples2.append(synth + snare)
        loop2 = self._create_stereo_sound(np.concatenate(samples2) * 0.32)
        if loop2:
            self.bgm_loops[2] = loop2

        # Intensitas 3: Intens (Tempo ~155 BPM, ritme sinkopasi dan kick arcade)
        bpm3 = 155
        beat_len3 = int(sr * (60.0 / bpm3) / 2)
        bass_seq3 = [146.83, 174.61, 220.00, 261.63, 196.00, 220.00, 164.81, 146.83]
        samples3 = []
        for i, f in enumerate(bass_seq3):
            t = np.linspace(0, beat_len3 / sr, beat_len3, endpoint=False)
            lead = (np.sign(np.sin(2 * np.pi * f * t)) * 0.2 + np.sin(2 * np.pi * f * t) * 0.5) * np.exp(-t * 9)
            kick = (np.sin(2 * np.pi * np.linspace(160, 40, len(t)) * t) * 0.4 * np.exp(-t * 20)) if (i % 2 == 0) else 0
            samples3.append(lead + kick)
        loop3 = self._create_stereo_sound(np.concatenate(samples3) * 0.36)
        if loop3:
            self.bgm_loops[3] = loop3

        # Intensitas 4: Ekstrem (Tempo ~180 BPM, arpeggio chiptune cepat)
        bpm4 = 180
        beat_len4 = int(sr * (60.0 / bpm4) / 4)  # Sixteenth note
        fast_seq = [
            261.63, 329.63, 392.00, 523.25,
            392.00, 329.63, 261.63, 196.00,
            220.00, 261.63, 329.63, 440.00,
            392.00, 329.63, 293.66, 261.63,
        ]
        samples4 = []
        for i, f in enumerate(fast_seq):
            t = np.linspace(0, beat_len4 / sr, beat_len4, endpoint=False)
            pulse = np.sin(2 * np.pi * f * t) * np.exp(-t * 12)
            kick = (np.sin(2 * np.pi * np.linspace(180, 50, len(t)) * t) * 0.3 * np.exp(-t * 25)) if (i % 4 == 0) else 0
            samples4.append(pulse + kick)
        loop4 = self._create_stereo_sound(np.concatenate(samples4) * 0.38)
        if loop4:
            self.bgm_loops[4] = loop4

    def play_sfx(self, sound_name: str) -> None:
        """Memutar efek suara instan jika audio aktif."""
        if not self.enabled or sound_name not in self.sfx_sounds:
            return
        try:
            snd = self.sfx_sounds[sound_name]
            snd.set_volume(self.sfx_volume)
            snd.play()
        except Exception:
            pass

    def set_bgm_intensity(self, intensity: int) -> None:
        """Mengatur tingkat intensitas musik in-game (1 s.d 4)."""
        if not self.enabled:
            return

        clamped = max(1, min(4, intensity))
        if clamped == self.current_intensity:
            return

        self.current_intensity = clamped
        loop = self.bgm_loops.get(clamped)
        if loop is None:
            return

        try:
            # Gunakan channel khusus 0 untuk musik latar
            bgm_channel = pygame.mixer.Channel(0)
            loop.set_volume(self.bgm_volume)
            bgm_channel.play(loop, loops=-1)
            self.active_bgm_channel = bgm_channel
        except Exception:
            pass

    def stop_bgm(self) -> None:
        """Menghentikan pemutaran musik latar."""
        self.current_intensity = 0
        if not self.enabled:
            return
        try:
            pygame.mixer.Channel(0).stop()
        except Exception:
            pass

    def toggle_sound(self) -> bool:
        """Menghidupkan atau mematikan seluruh audio permainan."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop_bgm()
        else:
            if self.current_intensity > 0:
                self.set_bgm_intensity(self.current_intensity)
        return self.enabled
