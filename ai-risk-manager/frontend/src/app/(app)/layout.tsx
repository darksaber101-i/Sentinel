import Navbar from "@/components/Navbar";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Navbar />
      <main className="ml-56 min-h-screen p-8">{children}</main>
    </>
  );
}
