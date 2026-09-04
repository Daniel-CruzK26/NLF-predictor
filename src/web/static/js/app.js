/**
 * NFL Gridiron AI Predictor - Dashboard Logic & Interactivity
 */

let allPredictions = [];
let allTeams = [];
let modelBenchmarks = null;
let activeFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadDashboardData();
    initSimulator();
    loadRosters();
    initRosterManager();
    initMonteCarloView();
});

// --------------------------------------------------------------------------
// Navigation & Tabs
// --------------------------------------------------------------------------
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetView = btn.dataset.view;
            
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.view-section').forEach(sec => {
                sec.classList.remove('active');
            });

            const activeSec = document.getElementById(`view-${targetView}`);
            if (activeSec) {
                activeSec.classList.add('active');
            }
        });
    });

    // Matchup Filter Chips
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(fBtn => {
        fBtn.addEventListener('click', () => {
            filterButtons.forEach(b => b.classList.remove('active', 'active-val'));
            fBtn.classList.add(fBtn.dataset.filter === 'value' ? 'active-val' : 'active');
            activeFilter = fBtn.dataset.filter;
            renderMatchupCards();
        });
    });
}

// --------------------------------------------------------------------------
// Data Fetching
// --------------------------------------------------------------------------
async function loadDashboardData() {
    try {
        const [predsRes, teamsRes, modelsRes] = await Promise.all([
            fetch('/api/predictions?season=2026&week=1'),
            fetch('/api/teams'),
            fetch('/api/models')
        ]);

        const predsData = await predsRes.json();
        const teamsData = await teamsRes.json();
        const modelsData = await modelsRes.json();

        allPredictions = predsData.predictions || [];
        allTeams = teamsData.teams || [];
        modelBenchmarks = modelsData;

        updateHeroKPIs();
        renderMatchupCards();
        renderTeamRankingsTable();
        renderBenchmarkTable();
        populateSimulatorDropdowns();

        const mcSelect = document.getElementById('mc-game-select');
        if (mcSelect && allPredictions.length > 0) {
            mcSelect.innerHTML = '';
            allPredictions.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.game_id;
                opt.textContent = `${p.matchup} (${p.gameday ? p.gameday.slice(5, 10) : ''})`;
                mcSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Error loading dashboard data:", err);
    }
}

// --------------------------------------------------------------------------
// Hero KPIs
// --------------------------------------------------------------------------
function updateHeroKPIs() {
    const valuePicks = allPredictions.filter(p => p.confidence !== 'NO VALUE');
    document.getElementById('kpi-value-picks').textContent = valuePicks.length;
    
    if (valuePicks.length > 0) {
        const avgEdge = valuePicks.reduce((acc, p) => acc + Math.abs(p.edge), 0) / valuePicks.length;
        document.getElementById('kpi-avg-edge').textContent = `${avgEdge.toFixed(1)} pts`;
        
        const topPick = [...valuePicks].sort((a, b) => Math.abs(b.edge) - Math.abs(a.edge))[0];
        document.getElementById('kpi-top-pick').textContent = topPick.recommendation;
    }
}

// --------------------------------------------------------------------------
// Render Matchup Cards
// --------------------------------------------------------------------------
function renderMatchupCards() {
    const container = document.getElementById('matchups-container');
    container.innerHTML = '';

    let filtered = allPredictions;
    if (activeFilter === 'value') {
        filtered = allPredictions.filter(p => p.confidence !== 'NO VALUE');
    } else if (activeFilter === '3star') {
        filtered = allPredictions.filter(p => p.confidence.includes('3★'));
    } else if (activeFilter === '2star') {
        filtered = allPredictions.filter(p => p.confidence.includes('2★'));
    } else if (activeFilter === '1star') {
        filtered = allPredictions.filter(p => p.confidence.includes('1★'));
    }

    filtered.forEach(p => {
        const card = document.createElement('div');
        card.className = 'match-card';
        card.onclick = () => openMatchupModal(p);

        // Confidence Class
        let confClass = 'conf-none';
        if (p.confidence.includes('3★')) confClass = 'conf-3star';
        else if (p.confidence.includes('2★')) confClass = 'conf-2star';
        else if (p.confidence.includes('1★')) confClass = 'conf-1star';

        const isNoEdge = p.confidence === 'NO VALUE';
        const vSpreadStr = p.vegas_spread > 0 ? `-${p.vegas_spread.toFixed(1)}` : `+${Math.abs(p.vegas_spread).toFixed(1)}`;
        const mSpreadStr = p.model_spread > 0 ? `-${p.model_spread.toFixed(1)}` : `+${Math.abs(p.model_spread).toFixed(1)}`;

        card.innerHTML = `
            <div class="match-header">
                <span class="match-date">📅 ${p.gameday ? p.gameday.slice(0, 10) : 'Sep 2026'}</span>
                <span class="conf-tag ${confClass}">${p.confidence}</span>
            </div>
            <div class="teams-row">
                <div class="team-box away">
                    <span class="team-abbr">${p.away_team}</span>
                    <span class="team-qb">👤 ${p.away_qb || 'QB'}</span>
                </div>
                <div class="vs-badge">AT</div>
                <div class="team-box home">
                    <span class="team-abbr">${p.home_team}</span>
                    <span class="team-qb">👤 ${p.home_qb || 'QB'}</span>
                </div>
            </div>
            <div class="prob-bar-wrapper">
                <div class="prob-labels">
                    <span>${p.away_win_prob}%</span>
                    <span>Win Prob</span>
                    <span>${p.home_win_prob}%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill-home" style="width: ${p.home_win_prob}%"></div>
                </div>
            </div>
            <div class="odds-box">
                <div class="odds-col">
                    <span>Vegas Line</span>
                    <strong>${vSpreadStr}</strong>
                </div>
                <div class="odds-col">
                    <span>Model Proj</span>
                    <strong>${mSpreadStr}</strong>
                </div>
                <div class="odds-col edge-col">
                    <span>Edge</span>
                    <strong>${Math.abs(p.edge).toFixed(1)} pts</strong>
                </div>
            </div>
            <div class="rec-footer ${isNoEdge ? 'no-edge' : ''}">
                <span class="rec-action">${p.recommendation}</span>
                <span class="rec-stake">${p.kelly_stake_pct > 0 ? `Stake: ${p.kelly_stake_pct}%` : 'No Bet'}</span>
            </div>
        `;
        container.appendChild(card);
    });
}

// --------------------------------------------------------------------------
// Matchup Detail Modal
// --------------------------------------------------------------------------
function openMatchupModal(p) {
    const modal = document.getElementById('matchup-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');

    title.textContent = `${p.away_team} at ${p.home_team} — Deep Dive Analysis`;

    body.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
            <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <h4 style="color: var(--accent-cyan); font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase;">Matchup Projections</h4>
                <p><strong>Vegas Spread:</strong> ${p.home_team} by ${p.vegas_spread.toFixed(1)} pts</p>
                <p><strong>AI Model Spread:</strong> ${p.home_team} by ${p.model_spread.toFixed(1)} pts</p>
                <p><strong>Market Edge:</strong> <span style="color: var(--accent-emerald); font-weight: bold;">${p.edge > 0 ? '+' : ''}${p.edge.toFixed(2)} pts</span></p>
                <p><strong>Win Probability:</strong> ${p.home_team} ${p.home_win_prob}% | ${p.away_team} ${p.away_win_prob}%</p>
            </div>
            <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <h4 style="color: var(--accent-emerald); font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase;">Betting Recommendation</h4>
                <p style="font-size: 1.15rem; font-weight: 800; color: var(--accent-emerald);">${p.recommendation}</p>
                <p><strong>Confidence Tier:</strong> ${p.confidence}</p>
                <p><strong>Kelly Allocation:</strong> ${p.kelly_stake_pct}% of total bankroll (1/4 Kelly)</p>
            </div>
        </div>
        <div style="background: rgba(14,21,38,0.7); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom:1rem;">
            <h4 style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">AI Analytical Drivers</h4>
            <p style="color: var(--text-primary); font-size: 0.95rem;">💡 ${p.key_drivers || 'Statistical equilibrium.'}</p>
        </div>
        <button class="btn-primary" style="width:100%;" onclick="launchMonteCarloFromModal('${p.game_id}')">
            🎲 Ver Simulación Monte Carlo (10,000 Iteraciones)
        </button>
    `;

    modal.classList.add('active');
}

function launchMonteCarloFromModal(gameId) {
    closeModal();
    const mcNav = document.querySelector('.nav-btn[data-view="monte-carlo"]');
    if (mcNav) mcNav.click();
    const mcSelect = document.getElementById('mc-game-select');
    if (mcSelect) {
        mcSelect.value = gameId;
        runMonteCarloSimulation(gameId);
    }
}

function closeModal() {
    document.getElementById('matchup-modal').classList.remove('active');
}

// --------------------------------------------------------------------------
// Team Power Rankings Table
// --------------------------------------------------------------------------
function renderTeamRankingsTable() {
    const tbody = document.getElementById('teams-tbody');
    tbody.innerHTML = '';

    allTeams.forEach(t => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="table-rank">#${t.rank}</td>
            <td class="table-team"><strong>${t.abbr}</strong> <span style="font-weight:400; color:var(--text-muted); font-size:0.8rem;">${t.name}</span></td>
            <td><span class="badge-green" style="background:rgba(56,189,248,0.15); color:var(--accent-cyan);">👤 ${t.starting_qb}</span></td>
            <td>${t.division}</td>
            <td class="mono-cell" style="color: var(--accent-emerald); font-weight: 800; font-size: 1rem;">${t.power_rating}</td>
            <td class="mono-cell">${t.adj_off_pass_epa > 0 ? '+' : ''}${t.adj_off_pass_epa}</td>
            <td class="mono-cell">${t.adj_def_pass_epa_allowed > 0 ? '+' : ''}${t.adj_def_pass_epa_allowed}</td>
            <td class="mono-cell" style="color: ${t.net_adj_epa > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-weight:700;">
                ${t.net_adj_epa > 0 ? '+' : ''}${t.net_adj_epa}
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// --------------------------------------------------------------------------
// Model Accuracy & Benchmark Table
// --------------------------------------------------------------------------
function renderBenchmarkTable() {
    if (!modelBenchmarks) return;

    const tbody = document.getElementById('models-tbody');
    tbody.innerHTML = '';

    modelBenchmarks.models.forEach(m => {
        const tr = document.createElement('tr');
        const isBest = m.name.includes('LightGBM');
        tr.innerHTML = `
            <td style="font-weight: 700; ${isBest ? 'color: var(--accent-emerald);' : ''}">${m.name}</td>
            <td><span class="badge-green">${m.type}</span></td>
            <td class="mono-cell">${m.mae.toFixed(3)} pts</td>
            <td class="mono-cell">${m.rmse.toFixed(3)}</td>
            <td class="mono-cell">${m.su_win_pct ? m.su_win_pct.toFixed(2) + '%' : '—'}</td>
            <td class="mono-cell" style="font-weight: 800; color: ${m.ats_pct > 52.4 ? 'var(--accent-emerald)' : 'inherit'};">${m.ats_pct ? m.ats_pct.toFixed(2) + '%' : '—'}</td>
        `;
        tbody.appendChild(tr);
    });

    const featContainer = document.getElementById('features-container');
    featContainer.innerHTML = '';
    modelBenchmarks.top_features.forEach(f => {
        const item = document.createElement('div');
        item.style.cssText = 'display:flex; justify-content:space-between; padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.88rem;';
        item.innerHTML = `
            <span>${f.feature}</span>
            <span class="mono-cell" style="color:var(--accent-cyan); font-weight:bold;">${(f.importance * 100).toFixed(1)}%</span>
        `;
        featContainer.appendChild(item);
    });
}

// --------------------------------------------------------------------------
// Active Rosters & QB Manager
// --------------------------------------------------------------------------
let activeStartersMap = {};

async function loadRosters() {
    try {
        const res = await fetch('/api/rosters');
        const data = await res.json();
        activeStartersMap = data.starters || {};
        renderRostersGrid();
    } catch (e) {
        console.error("Error loading rosters:", e);
    }
}

function renderRostersGrid() {
    const grid = document.getElementById('rosters-grid');
    if (!grid) return;
    grid.innerHTML = '';

    allTeams.forEach(t => {
        const currentStarter = activeStartersMap[t.abbr] || t.starting_qb || '';
        const card = document.createElement('div');
        card.style.cssText = 'background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1rem;';
        card.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                <strong style="font-size:1.1rem; color:var(--text-primary);">${t.abbr}</strong>
                <span style="font-size:0.75rem; color:var(--text-muted);">${t.division}</span>
            </div>
            <div style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:0.4rem;">${t.name}</div>
            <div class="form-group">
                <label style="font-size:0.7rem;">QB Titular Activo</label>
                <input type="text" class="form-control roster-input" data-team="${t.abbr}" value="${currentStarter}" style="font-size:0.88rem; padding:0.45rem 0.65rem;">
            </div>
        `;
        grid.appendChild(card);
    });
}

function initRosterManager() {
    const saveBtn = document.getElementById('btn-save-rosters');
    const syncBtn = document.getElementById('btn-sync-espn');
    const syncFullBtn = document.getElementById('btn-sync-full-rosters');

    if (syncFullBtn) {
        syncFullBtn.addEventListener('click', async () => {
            syncFullBtn.textContent = '⏳ Descargando 32 Rosters de ESPN...';
            syncFullBtn.disabled = true;

            try {
                const res = await fetch('/api/rosters/sync-full-espn', { method: 'POST' });
                const data = await res.json();
                await loadDashboardData();
                await loadRosters();
                alert(`✅ ¡Rosters completos actualizados para los ${data.teams_count} equipos directamente desde ESPN!`);
            } catch (e) {
                console.error("Error syncing full rosters:", e);
                alert('❌ Error al sincronizar plantillas completas con ESPN.');
            } finally {
                syncFullBtn.textContent = '⚡ Sincronizar Rosters Completos (ESPN)';
                syncFullBtn.disabled = false;
            }
        });
    }

    if (syncBtn) {
        syncBtn.addEventListener('click', async () => {
            syncBtn.textContent = '⏳ Consultando API de ESPN...';
            syncBtn.disabled = true;

            try {
                const res = await fetch('/api/rosters/sync-espn', { method: 'POST' });
                const data = await res.json();
                activeStartersMap = data.starters || {};
                renderRostersGrid();
                await loadDashboardData();
                alert('✅ ¡Sincronizado exitosamente con la API oficial de ESPN en vivo!');
            } catch (e) {
                console.error("Error syncing with ESPN:", e);
                alert('❌ Error al sincronizar con ESPN.');
            } finally {
                syncBtn.textContent = '🔄 Sincronizar con ESPN';
                syncBtn.disabled = false;
            }
        });
    }

    if (saveBtn) {
        saveBtn.addEventListener('click', async () => {
            const inputs = document.querySelectorAll('.roster-input');
            const payload = {};
            inputs.forEach(inp => {
                payload[inp.dataset.team] = inp.value.trim();
            });

            saveBtn.textContent = 'Guardando y reentrenando...';
            saveBtn.disabled = true;

            try {
                await fetch('/api/rosters', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                await loadDashboardData();
                alert('✅ ¡Plantillas actualizadas y modelos recalculados con éxito!');
            } catch (e) {
                console.error("Error saving rosters:", e);
                alert('❌ Error al guardar plantillas.');
            } finally {
                saveBtn.textContent = '💾 Guardar y Recalcular';
                saveBtn.disabled = false;
            }
        });
    }
}

// --------------------------------------------------------------------------
// Custom Simulator Form
// --------------------------------------------------------------------------
function populateSimulatorDropdowns() {
    const homeSelect = document.getElementById('sim-home-team');
    const awaySelect = document.getElementById('sim-away-team');
    if (!homeSelect || !awaySelect) return;

    homeSelect.innerHTML = '';
    awaySelect.innerHTML = '';

    allTeams.forEach(t => {
        const optH = document.createElement('option');
        optH.value = t.abbr;
        optH.textContent = `${t.abbr} - ${t.name}`;
        homeSelect.appendChild(optH);

        const optA = document.createElement('option');
        optA.value = t.abbr;
        optA.textContent = `${t.abbr} - ${t.name}`;
        awaySelect.appendChild(optA);
    });

    homeSelect.value = 'KC';
    awaySelect.value = 'BAL';
}

function initSimulator() {
    const simBtn = document.getElementById('btn-run-sim');
    if (!simBtn) return;

    simBtn.addEventListener('click', async () => {
        const homeTeam = document.getElementById('sim-home-team').value;
        const awayTeam = document.getElementById('sim-away-team').value;
        const vegasSpread = parseFloat(document.getElementById('sim-vegas-spread').value) || 0.0;
        const wind = parseFloat(document.getElementById('sim-wind').value) || 7.0;
        const temp = parseFloat(document.getElementById('sim-temp').value) || 70.0;
        const isDome = parseInt(document.getElementById('sim-dome').value) || 0;
        const restDiff = parseInt(document.getElementById('sim-rest').value) || 0;

        simBtn.textContent = 'Calculando simulación con IA...';
        simBtn.disabled = true;

        try {
            const res = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    home_team: homeTeam,
                    away_team: awayTeam,
                    vegas_spread: vegasSpread,
                    wind_speed: wind,
                    temperature: temp,
                    is_dome: isDome,
                    rest_diff: restDiff
                })
            });

            const result = await res.json();
            renderSimulationResult(result);
        } catch (err) {
            console.error("Simulation error:", err);
        } finally {
            simBtn.textContent = 'Simular Partido con IA';
            simBtn.disabled = false;
        }
    });
}

function renderSimulationResult(r) {
    const resBox = document.getElementById('sim-result-box');
    resBox.style.display = 'block';
    resBox.scrollIntoView({ behavior: 'smooth' });

    resBox.innerHTML = `
        <div style="background: rgba(18,26,47,0.9); border: 1px solid var(--border-bright); border-radius: 16px; padding: 2rem; box-shadow: var(--shadow-card), var(--glow-cyan);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1rem;">
                <h3 style="font-family: var(--font-display); font-size: 1.4rem;">Simulación: ${r.away_team} @ ${r.home_team}</h3>
                <span class="conf-tag conf-3star">${r.confidence}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; text-align: center; margin-bottom: 1.5rem;">
                <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 10px;">
                    <span style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Spread Proyectado</span>
                    <h2 style="font-family:var(--font-mono); color:var(--accent-cyan); font-size: 1.7rem;">${r.home_team} ${r.model_spread > 0 ? '-' : '+'}${Math.abs(r.model_spread)}</h2>
                </div>
                <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 10px;">
                    <span style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Prob. de Victoria</span>
                    <h2 style="font-family:var(--font-mono); color:var(--text-primary); font-size: 1.7rem;">${r.home_win_prob}% / ${r.away_win_prob}%</h2>
                </div>
                <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 10px;">
                    <span style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Edge de Mercado</span>
                    <h2 style="font-family:var(--font-mono); color:var(--accent-emerald); font-size: 1.7rem;">${Math.abs(r.edge).toFixed(1)} pts</h2>
                </div>
            </div>
            <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 1rem 1.5rem; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Recomendación del Modelo</span>
                    <h4 style="color:var(--accent-emerald); font-size:1.2rem; font-weight:800;">${r.recommendation}</h4>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Kelly Stake (1/4)</span>
                    <h4 style="font-family:var(--font-mono); color:var(--text-primary); font-size:1.2rem;">${r.kelly_stake_pct}%</h4>
                </div>
            </div>
        </div>
    `;
}

// --------------------------------------------------------------------------
// Monte Carlo Simulation (10,000 Iterations) & Star Player Management
// --------------------------------------------------------------------------
let starsCatalog = {};
let starInjuries = {};

function initMonteCarloView() {
    const runBtn = document.getElementById('btn-run-mc');
    const select = document.getElementById('mc-game-select');

    if (runBtn) {
        runBtn.addEventListener('click', () => {
            const gameId = select ? select.value : (allPredictions[0] ? allPredictions[0].game_id : null);
            if (gameId) {
                runMonteCarloSimulation(gameId);
            }
        });
    }

    // Auto-run when switching to Monte Carlo tab if first time
    const mcNav = document.querySelector('.nav-btn[data-view="monte-carlo"]');
    if (mcNav) {
        mcNav.addEventListener('click', () => {
            const resContainer = document.getElementById('mc-result-container');
            if (resContainer && resContainer.style.display === 'none') {
                const gameId = select ? select.value : (allPredictions[0] ? allPredictions[0].game_id : null);
                if (gameId) runMonteCarloSimulation(gameId);
            }
        });
    }
}

async function loadStarsData() {
    try {
        const res = await fetch('/api/stars');
        const data = await res.json();
        starsCatalog = data.stars || {};
        starInjuries = data.injuries || {};
    } catch (e) {
        console.error("Error loading stars:", e);
    }
}

async function runMonteCarloSimulation(gameId) {
    const runBtn = document.getElementById('btn-run-mc');
    if (runBtn) {
        runBtn.textContent = '⏳ Simulando 10k...';
        runBtn.disabled = true;
    }

    try {
        await loadStarsData();
        const res = await fetch(`/api/monte-carlo/${gameId}`);
        const data = await res.json();
        renderMonteCarloResult(data, gameId);
    } catch (err) {
        console.error("Error running Monte Carlo simulation:", err);
        alert("Error ejecutando la simulación Monte Carlo.");
    } finally {
        if (runBtn) {
            runBtn.textContent = '🎲 Simular 10k';
            runBtn.disabled = false;
        }
    }
}

function renderMonteCarloResult(data, gameId) {
    const container = document.getElementById('mc-result-container');
    container.style.display = 'block';

    // Summary KPIs
    document.getElementById('mc-score-val').textContent = `${data.away_team} ${data.mean_away_score} - ${data.mean_home_score} ${data.home_team}`;
    document.getElementById('mc-win-prob-sub').textContent = `Probabilidad: ${data.home_team} ${data.home_win_prob}% | ${data.away_team} ${data.away_win_prob}%`;
    
    document.getElementById('mc-spread-val').textContent = `${data.home_team} ${data.mean_spread > 0 ? '-' : '+'}${Math.abs(data.mean_spread).toFixed(1)} pts`;
    document.getElementById('mc-spread-sub').textContent = `Línea Vegas: ${data.home_team} ${data.vegas_spread > 0 ? '-' : '+'}${Math.abs(data.vegas_spread).toFixed(1)} (Cover: ${data.spread_cover_prob_home}%)`;

    document.getElementById('mc-total-pick').textContent = data.over_under_pick;
    document.getElementById('mc-total-sub').textContent = `Media total: ${data.mean_total} pts | Línea Vegas: ${data.vegas_total} pts`;

    const starAdj = data.star_impact ? data.star_impact.star_spread_adjustment : 0.0;
    const starTotalAdj = data.star_impact ? data.star_impact.star_total_adjustment : 0.0;
    document.getElementById('mc-star-impact-val').textContent = `${starAdj > 0 ? '+' : ''}${starAdj.toFixed(1)} pts`;
    document.getElementById('mc-star-impact-sub').textContent = `Impacto Total: ${starTotalAdj > 0 ? '+' : ''}${starTotalAdj.toFixed(1)} pts`;

    // Over/Under Card
    document.getElementById('mc-over-prob').textContent = `${data.over_prob}%`;
    document.getElementById('mc-under-prob').textContent = `${data.under_prob}%`;
    document.getElementById('mc-teaser-info').innerHTML = `
        <strong>Teasers (+6.0 pts):</strong><br>
        • ${data.home_team} ${data.vegas_spread > 0 ? '-' : '+'}${(Math.abs(data.vegas_spread) - 6.0).toFixed(1)} ➔ <strong>${data.teaser_prob_home_6pts}%</strong> cubre<br>
        • ${data.away_team} ${data.vegas_spread < 0 ? '-' : '+'}${(Math.abs(data.vegas_spread) + 6.0).toFixed(1)} ➔ <strong>${data.teaser_prob_away_6pts}%</strong> cubre
    `;

    // Key Numbers
    const knGrid = document.getElementById('mc-key-numbers-grid');
    knGrid.innerHTML = '';
    const topKeyNums = ['3', '7', '6', '10', '14', '1', '4', '2'];
    topKeyNums.forEach(num => {
        const prob = data.key_numbers_probs[num] || 0.0;
        const box = document.createElement('div');
        box.style.cssText = 'background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06); padding: 0.5rem; border-radius: 6px; text-align: center;';
        box.innerHTML = `
            <div style="font-size: 0.72rem; color: var(--text-muted);">Margen ${num}</div>
            <div style="font-size: 1rem; font-weight: 800; color: ${prob >= 8.0 ? 'var(--accent-emerald)' : 'var(--text-primary)'};">${prob}%</div>
        `;
        knGrid.appendChild(box);
    });

    // Top Exact Scores
    const scoresList = document.getElementById('mc-top-scores-list');
    scoresList.innerHTML = '';
    data.top_exact_scores.forEach(s => {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex; justify-content:space-between; align-items:center; background: rgba(0,0,0,0.25); padding: 0.45rem 0.8rem; border-radius: 6px; font-size: 0.82rem;';
        row.innerHTML = `
            <span style="font-family:var(--font-mono); font-weight:700;">${s.score}</span>
            <span class="badge-green" style="font-size:0.75rem;">${s.prob}% freq</span>
        `;
        scoresList.appendChild(row);
    });

    // Star Players Panel (Home & Away)
    const starsContainer = document.getElementById('mc-stars-container');
    starsContainer.innerHTML = '';

    const renderTeamStars = (teamAbbr, label) => {
        const players = starsCatalog[teamAbbr] || [];
        const teamBox = document.createElement('div');
        teamBox.style.cssText = 'background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 0.8rem;';
        
        let html = `<h4 style="font-size:0.85rem; color:var(--accent-cyan); margin-bottom:0.6rem; text-transform:uppercase;">${label} (${teamAbbr})</h4>`;
        if (players.length === 0) {
            html += `<p style="font-size:0.75rem; color:var(--text-muted);">Sin jugadores estelares registrados.</p>`;
        } else {
            players.forEach(p => {
                const currentStatus = (starInjuries[p.name] || 'ACTIVE').toUpperCase();
                html += `
                    <div style="display:flex; justify-content:space-between; align-items:center; padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <div>
                            <strong style="font-size:0.82rem;">${p.name}</strong> 
                            <span style="font-size:0.7rem; color:var(--text-muted);">(${p.pos} • ${p.value_pts} pts)</span>
                        </div>
                        <select class="form-control star-status-select" data-player="${p.name}" style="width: 120px; font-size: 0.72rem; padding: 0.25rem 0.4rem;">
                            <option value="ACTIVE" ${currentStatus === 'ACTIVE' ? 'selected' : ''}>🟢 Activo</option>
                            <option value="QUESTIONABLE" ${currentStatus === 'QUESTIONABLE' ? 'selected' : ''}>🟡 Cuestionable</option>
                            <option value="OUT" ${currentStatus === 'OUT' ? 'selected' : ''}>🔴 Fuera (Out)</option>
                        </select>
                    </div>
                `;
            });
        }
        teamBox.innerHTML = html;
        starsContainer.appendChild(teamBox);
    };

    renderTeamStars(data.home_team, 'Local');
    renderTeamStars(data.away_team, 'Visitante');

    // Add event listeners on status selects to dynamically update injury and re-run!
    document.querySelectorAll('.star-status-select').forEach(sel => {
        sel.addEventListener('change', async (e) => {
            const pName = e.target.dataset.player;
            const newStatus = e.target.value;
            try {
                await fetch('/api/stars/injury', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ player_name: pName, status: newStatus }),
                });
                starInjuries[pName] = newStatus;
                // Instant re-simulation!
                await runMonteCarloSimulation(gameId);
            } catch (err) {
                console.error("Error updating injury status:", err);
            }
        });
    });
}

