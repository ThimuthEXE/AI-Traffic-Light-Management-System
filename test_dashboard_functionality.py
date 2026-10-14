import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app import app

def test_dashboard():
    print("==================================================")
    print("Testing AI Traffic Light Management Dashboard & API")
    print("==================================================")
    
    with TestClient(app) as client:
        # 1. Root redirect and /dashboard trailing slash redirect
        res_root = client.get('/', follow_redirects=False)
        print(f"[TEST 1a] Root redirect: Status {res_root.status_code}, Location: {res_root.headers.get('location')}")
        assert res_root.status_code in [302, 307] and res_root.headers.get('location') == '/dashboard/', f"Expected 307 to /dashboard/, got {res_root.status_code}"

        res_dash = client.get('/dashboard', follow_redirects=False)
        print(f"[TEST 1b] /dashboard redirect: Status {res_dash.status_code}, Location: {res_dash.headers.get('location')}")
        assert res_dash.status_code in [302, 307] and res_dash.headers.get('location') == '/dashboard/', f"Expected 307 to /dashboard/, got {res_dash.status_code}"

        # 2. Static Dashboard HTML & Asset Resolution
        res = client.get('/dashboard/')
        print(f"[TEST 2a] Static Dashboard HTML: Status {res.status_code}, Length: {len(res.text)} bytes")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "<!DOCTYPE html>" in res.text, "index.html missing DOCTYPE"
        assert '<base href="/dashboard/">' in res.text, "index.html missing <base href='/dashboard/'>"

        # 2b. Static Assets at /dashboard/ and root /
        assets_to_test = [
            '/dashboard/app.js', '/app.js',
            '/dashboard/styles.css', '/styles.css',
            '/dashboard/tailwind.min.js', '/tailwind.min.js',
            '/dashboard/chart.min.js', '/chart.min.js',
            '/dashboard/lucide.min.js', '/lucide.min.js'
        ]
        for asset in assets_to_test:
            res_asset = client.get(asset)
            assert res_asset.status_code == 200, f"Asset {asset} returned {res_asset.status_code}"
        print(f"[TEST 2b] All Offline Static Assets (JS, CSS, Icons) verified at /dashboard/ and root / (All 200 OK)")

        # 3. System API & Status
        res = client.get('/api')
        assert res.status_code == 200
        print(f"[TEST 3] /api info: {res.json().get('system')}")

        res = client.get('/api/status')
        assert res.status_code == 200
        status_data = res.json()
        print(f"[TEST 4] /api/status: Status={status_data.get('status')}, Controller={status_data.get('active_controller')}, DB={status_data.get('database_mode')}")

        # 4. Intersections
        res = client.get('/api/intersections')
        assert res.status_code == 200
        print(f"[TEST 5] /api/intersections: Listed {len(res.json())} intersection(s)")

        res = client.get('/api/intersections/INT-01/live')
        assert res.status_code == 200
        live = res.json()
        print(f"[TEST 6] /api/intersections/INT-01/live: Mode={live.get('control_mode')}, Phase={live.get('active_phase', {}).get('phase_name')}")

        # 5. Control Mode Switch
        res = client.post('/api/intersections/INT-01/control', json={'mode': 1})
        assert res.status_code == 200
        print(f"[TEST 7] Switch to Fixed-Time: {res.json()}")

        res = client.post('/api/intersections/INT-01/control', json={'mode': 2})
        assert res.status_code == 200
        print(f"[TEST 8] Switch to AI Adaptive: {res.json()}")

        # 6. Emergency Preemption
        res = client.post('/api/intersections/INT-01/control', json={'mode': 2, 'emergency_axis': 'EW'})
        assert res.status_code == 200
        print(f"[TEST 9] Emergency Preemption Override: {res.json()}")

        # 7. Density Adjustment
        res = client.post('/api/intersections/INT-01/density', json={'direction': 'N', 'spawn_interval_sec': 1.2})
        assert res.status_code == 200
        print(f"[TEST 10] Density Update: {res.json()}")

        # 7b. Simulation Sync Handshake (delivers density & emergency to Pygame)
        sim_payload = {
            'intersection_id': 'INT-01',
            'active_mode': 2,
            'control_mode_name': 'AI_CYCLE_ADAPTIVE',
            'signals': {'North': 'RED', 'South': 'RED', 'East': 'GREEN', 'West': 'GREEN'},
            'active_phase': {'phase_name': 'EW_THROUGH_GREEN', 'active_axis': 'EW', 'allocated_green': 20.0, 'countdown_seconds': 15},
            'approaches': {'North': {'spawn_interval_sec': 2.0, 'arrival_flow_vpm': 30.0, 'density_category': 'MED'}},
            'ai_decision': {},
            'metrics': {},
            'emergency_active': False
        }
        sync_res = client.post('/api/intersections/INT-01/sync', json=sim_payload)
        assert sync_res.status_code == 200
        sync_data = sync_res.json()
        assert sync_data.get('approach_intervals', {}).get('N') == 1.2
        assert sync_data.get('density_version', 0) > 0
        assert sync_data.get('emergency_trigger') == 'EW'
        assert sync_data.get('emergency_version', 0) > 0
        print(f"[TEST 10b] Simulation Sync Handshake: Versioned Density Update (N=1.2s) & Emergency Override (EW) verified!")

        # 8. ML Predictions
        res = client.get('/api/predictions/forecast?datetime_str=2026-08-25T08:00:00&weather=Clear')
        assert res.status_code == 200
        pred = res.json()
        print(f"[TEST 11] ML Forecast: Critical Approach={pred.get('critical_tidal_approach')}, Flows={pred.get('predicted_flows_vpm')}")

        res = client.get('/api/predictions/model-info')
        assert res.status_code == 200
        print(f"[TEST 12] ML Model Info: Status={res.json().get('status')}")

        # 9. Analytics
        res = client.get('/api/analytics/summary')
        assert res.status_code == 200
        print(f"[TEST 13] Analytics Summary: AWT Reduction={res.json().get('performance_comparison', {}).get('waiting_time_reduction_pct')}%")

        res = client.get('/api/analytics/cycles?limit=5')
        assert res.status_code == 200
        print(f"[TEST 14] Analytics Cycles: Retrieved {len(res.json())} logs")

        # 10. Real-Time WebSocket Telemetry
        print("[TEST 15] Connecting to WebSocket /ws/live-traffic...")
        with client.websocket_connect('/ws/live-traffic') as ws:
            frame = ws.receive_json()
            assert 'intersection_id' in frame
            assert 'signals' in frame
            assert 'active_phase' in frame
            print(f"  -> WebSocket Frame Received Successfully!")
            print(f"  -> Active Phase: {frame.get('active_phase', {}).get('phase_name')}")
            print(f"  -> Signals: {frame.get('signals')}")
            print(f"  -> Countdown: {frame.get('active_phase', {}).get('countdown_seconds')}s")

    print("\n==================================================")
    print("ALL 15 DASHBOARD FUNCTIONALITY TESTS PASSED (100% SUCCESS)")
    print("==================================================")

if __name__ == '__main__':
    test_dashboard()
