import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Leaderboard from "./pages/Leaderboard";
import RunDetail from "./pages/RunDetail";
import NewRun from "./pages/NewRun";
import MetricsGuide from "./pages/MetricsGuide";

export default function App() {
  return (
    <>
      <nav className="topnav">
        <div className="brand">
          Open<span>TS</span>Bench
        </div>
        <NavLink to="/leaderboard" className={({ isActive }) => `navlink${isActive ? " active" : ""}`}>
          排行榜
        </NavLink>
        <NavLink to="/new" className={({ isActive }) => `navlink${isActive ? " active" : ""}`}>
          发起实验
        </NavLink>
        <NavLink to="/metrics" className={({ isActive }) => `navlink${isActive ? " active" : ""}`}>
          指标说明
        </NavLink>
      </nav>
      <div className="page">
        <Routes>
          <Route path="/" element={<Navigate to="/leaderboard" replace />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
          <Route path="/new" element={<NewRun />} />
          <Route path="/metrics" element={<MetricsGuide />} />
        </Routes>
      </div>
    </>
  );
}
