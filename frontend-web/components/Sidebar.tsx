"use client";

import {
  ChartNoAxesCombined, Landmark, LayoutDashboard, ListOrdered, ShieldCheck, UserRound,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Home", icon: LayoutDashboard },
  { href: "/predict", label: "Single prediction", icon: UserRound },
  { href: "/batch", label: "Call list", icon: ListOrdered },
  { href: "/insights", label: "Insights", icon: ChartNoAxesCombined },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="bg-white border-[#dde4df] border-r border-solid flex flex-col items-start justify-between px-[18px] py-[28px] h-screen sticky top-0 w-[260px] shrink-0">
      <div className="flex flex-col gap-[38px] items-start w-full">
        <Link href="/" className="flex gap-[10px] items-center px-[8px] w-full">
          <div className="bg-[#176547] flex h-[36px] items-center justify-center rounded-[6px] shrink-0 w-[32px]">
            <Landmark size={20} className="text-white" />
          </div>
          <div className="flex flex-col items-start leading-[normal]">
            <p className="font-bold text-[#23362f] text-[21px]">Northbank</p>
            <p className="text-[#65736d] text-[12px]">Marketing workspace</p>
          </div>
        </Link>

        <div className="flex flex-col gap-[6px] items-start w-full">
          <p className="font-semibold text-[#8a958f] text-[11px] uppercase">Term deposits</p>
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link key={href} href={href}
                className={`flex gap-[12px] items-center p-[12px] rounded-[6px] w-full ${active ? "bg-[#eaf3ee]" : "bg-white hover:bg-[#f5f7f6]"}`}>
                <Icon size={20} className={active ? "text-[#176547]" : "text-[#65736d]"} />
                <p className={`text-[16px] ${active ? "font-semibold text-[#176547]" : "text-[#65736d]"}`}>{label}</p>
              </Link>
            );
          })}
        </div>
      </div>

      <div className="flex flex-col gap-[18px] items-start w-full">
        <div className="bg-[#f4f6f5] flex flex-col gap-[7px] items-start p-[14px] rounded-[6px] w-full">
          <ShieldCheck size={20} className="text-[#176547]" />
          <p className="font-semibold text-[#23362f] text-[14px] leading-[1.4]">Customer choice first</p>
          <p className="text-[#65736d] text-[13px] leading-[1.4]">
            Always review contact preferences before outreach.
          </p>
        </div>
        <div className="flex gap-[10px] items-center">
          <div className="bg-[#eef1ef] flex items-center justify-center rounded-full shrink-0 size-[32px]">
            <p className="text-[#65736d] text-[12px]">MT</p>
          </div>
          <p className="text-[#65736d] text-[14px]">Marketing team</p>
        </div>
      </div>
    </div>
  );
}
