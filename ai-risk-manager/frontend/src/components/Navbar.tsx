"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldAlert, LayoutDashboard, ListChecks, List, BarChart3, SlidersHorizontal, Bot } from "lucide-react";
import { clsx } from "clsx";
import { api } from "@/lib/api";

const GROUPS = [
  {
    label: "Operate",
    items: [
      { href: "/dashboard",     label: "Control Center", icon: LayoutDashboard },
      { href: "/review-queue",  label: "Review Queue",   icon: ListChecks, badge: true },
      { href: "/orders",        label: "Orders",         icon: List },
    ],
  },
  {
    label: "Understand",
    items: [
      { href: "/model-performance",   label: "Model Trust",     icon: BarChart3 },
      { href: "/threshold-simulator", label: "Policy Simulator",icon: SlidersHorizontal },
      { href: "/assistant",           label: "AI Assistant",    icon: Bot },
    ],
  },
];

export default function Navbar() {
  const path = usePathname();
  const [queueCount, setQueueCount] = useState<number | null>(null);

  useEffect(() => {
    api.getDashboardStats().then(d => setQueueCount(d.queue_count)).catch(() => {});
  }, []);

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-surface border-r border-border flex flex-col z-50">
      <Link href="/" className="flex items-center gap-2 px-5 py-5 border-b border-border hover:bg-card transition-colors">
        <ShieldAlert className="text-amber" size={22} />
        <div>
          <p className="text-sm font-semibold text-text-primary leading-tight">Sentinel</p>
          <p className="text-[10px] text-text-muted">Detect risk before loss</p>
        </div>
      </Link>
      <nav className="flex-1 py-4 px-3 space-y-5 overflow-y-auto">
        {GROUPS.map(group => (
          <div key={group.label}>
            <p className="px-3 mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">{group.label}</p>
            <div className="space-y-1">
              {group.items.map(({ href, label, icon: Icon, badge }) => {
                const active = path === href || path.startsWith(href + "/");
                return (
                  <Link
                    key={href}
                    href={href}
                    className={clsx(
                      "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-colors",
                      active
                        ? "bg-amber/10 text-amber font-medium"
                        : "text-text-secondary hover:bg-card hover:text-text-primary"
                    )}
                  >
                    <Icon size={16} />
                    <span className="flex-1">{label}</span>
                    {badge && queueCount != null && queueCount > 0 && (
                      <span className="text-[10px] font-semibold bg-amber-solid text-black rounded-full px-1.5 py-0.5 min-w-[18px] text-center">
                        {queueCount > 99 ? "99+" : queueCount}
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
